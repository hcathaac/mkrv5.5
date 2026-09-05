"""Reusable data, statistics and econometrics engine for the Streamlit app.

The module deliberately keeps UI code out of the calculations so that every
table can be tested, exported and reproduced from Python or R.
"""
from __future__ import annotations

import io
import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
import statsmodels.api as sm


MAX_DEPENDENT = 1000
MAX_INDEPENDENT = 1000

DOCUMENTATION_SHEET_TOKENS = (
    "readme", "read_me", "read me", "data_dictionary", "data dictionary",
    "dictionary", "codebook", "questionnaire_mapping", "questionnaire mapping",
    "crosswalk", "quality_control", "quality control", "metadata",
    "processing_log", "processing log", "maturity_code", "maturity code",
    "record_status", "record status",
)


def is_likely_analytical_frame(label: str, frame: pd.DataFrame) -> bool:
    """Separate probable observation tables from workbook documentation sheets."""
    folded = re.sub(r"\s+", " ", str(label).casefold().replace("-", "_")).strip()
    return not frame.empty and not any(token in folded for token in DOCUMENTATION_SHEET_TOKENS)


def safe_result_metric(result: Any, name: str) -> float:
    """Read an estimator metric without triggering unsupported cached properties."""
    try:
        value = getattr(result, name)
        return float(value) if value is not None else np.nan
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return np.nan


def safe_name(value: Any) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    text = re.sub(r"[^0-9A-Za-z_\u0370-\u03ff\u1f00-\u1fff]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_") or "variable"
    return text


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in columns:
        base = safe_name(raw)
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def tidy_frame(df: pd.DataFrame, normalise_columns: bool = False) -> pd.DataFrame:
    out = promote_embedded_header(df)
    if normalise_columns:
        out.columns = make_unique_columns(out.columns)
    else:
        out.columns = [str(c).strip() for c in out.columns]
        if len(set(out.columns)) != len(out.columns):
            out.columns = make_unique_columns(out.columns)
    for col in out.select_dtypes(include="object").columns:
        s = out[col].astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})
        parsed = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
        out[col] = parsed if parsed.notna().mean() >= .9 else s
    return out


def promote_embedded_header(df: pd.DataFrame) -> pd.DataFrame:
    """Promote row 1 when Excel contains a numeric index row above real names.

    The supplied 83-variable R&D workbook has exactly this structure: Excel's
    first row is 1..83 and its second row contains the analytical names.
    """
    if df.empty:
        return df.copy()
    current = [str(c).strip() for c in df.columns]
    numeric_headers = sum(bool(re.fullmatch(r"(?:Unnamed: )?\d+(?:\.0)?", c)) for c in current)
    first = df.iloc[0].astype("string")
    textual_first = float(first.str.contains(r"[A-Za-z\u0370-\u03ff]", regex=True, na=False).mean())
    unique_first = first.nunique(dropna=True) == len(first.dropna())
    if numeric_headers >= .7 * len(current) and textual_first >= .5 and unique_first:
        out = df.iloc[1:].copy().reset_index(drop=True)
        out.columns = [str(v).strip() for v in first]
        return out
    return df.copy()


def read_tabular_bytes(name: str, payload: bytes, all_sheets: bool = True) -> dict[str, pd.DataFrame]:
    """Read CSV/TSV/XLS/XLSX/Parquet/Feather/Arrow bytes and return named datasets."""
    lower = name.lower()
    src = io.BytesIO(payload)
    frames: dict[str, pd.DataFrame] = {}
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        book = pd.ExcelFile(src)
        sheets = book.sheet_names if all_sheets else book.sheet_names[:1]
        for sheet in sheets:
            frame = pd.read_excel(book, sheet_name=sheet)
            frames[f"{name} :: {sheet}"] = frame
    elif lower.endswith((".parquet", ".pq")):
        try:
            frames[name] = pd.read_parquet(src, engine="pyarrow")
        except Exception as exc:
            raise RuntimeError("Parquet input requires PyArrow; install the bundled v5.8.0 requirements.") from exc
    elif lower.endswith(".feather"):
        try:
            frames[name] = pd.read_feather(src)
        except Exception as exc:
            raise RuntimeError("Feather input requires PyArrow; install the bundled v5.8.0 requirements.") from exc
    elif lower.endswith(".arrow"):
        try:
            import pyarrow.ipc as ipc
            src.seek(0)
            reader = ipc.open_file(src)
            frames[name] = reader.read_all().to_pandas()
        except Exception:
            try:
                src.seek(0)
                import pyarrow.ipc as ipc
                reader = ipc.open_stream(src)
                frames[name] = reader.read_all().to_pandas()
            except Exception as exc:
                raise RuntimeError("Arrow IPC input requires PyArrow; install the bundled v5.8.0 requirements.") from exc
    elif lower.endswith(".tsv"):
        frames[name] = pd.read_csv(src, sep="\t", low_memory=False)
    elif lower.endswith(".csv"):
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1253", "latin1"):
            try:
                src.seek(0)
                frames[name] = pd.read_csv(src, encoding=encoding, sep=None, engine="python", low_memory=False)
                break
            except Exception as exc:  # pragma: no cover - depends on user input
                last_error = exc
        if not frames:
            raise ValueError(f"Could not read {name}: {last_error}")
    else:
        raise ValueError(f"Unsupported file type: {name}")
    return frames


def combine_frames(
    frames: Mapping[str, pd.DataFrame],
    mode: str,
    join_keys: Sequence[str] | None = None,
    join_how: str = "outer",
) -> pd.DataFrame:
    """Append rows, merge on keys, align columns by row order, or retain one dataset.

    Side-by-side alignment is deliberately explicit because it does not infer a
    relational key. Row 1 in each source is treated as the same observational
    unit, row 2 as the same unit, and so on. Unequal source lengths are retained
    through an outer index alignment and therefore create missing cells.
    """
    if not frames:
        return pd.DataFrame()
    if mode == "Combine columns side-by-side (by row order)":
        aligned: list[pd.DataFrame] = []
        used: set[str] = {"__row_position__"}
        for dataset_number, (_, frame) in enumerate(frames.items(), start=1):
            d = frame.copy().reset_index(drop=True)
            renamed: dict[str, str] = {}
            for column in d.columns:
                base = str(column)
                candidate = base
                if candidate in used:
                    candidate = f"{base}__d{dataset_number}"
                    duplicate_number = 2
                    while candidate in used:
                        candidate = f"{base}__d{dataset_number}_{duplicate_number}"
                        duplicate_number += 1
                renamed[column] = candidate
                used.add(candidate)
            aligned.append(d.rename(columns=renamed))
        result = pd.concat(aligned, axis=1, join="outer")
        result.insert(0, "__row_position__", np.arange(1, len(result) + 1))
        return result
    clean = []
    for label, frame in frames.items():
        d = frame.copy()
        d["__source_dataset__"] = label
        clean.append(d)
    if mode == "Append rows (union by column name)":
        return pd.concat(clean, ignore_index=True, sort=False)
    if mode == "Join datasets on key(s)":
        keys = list(join_keys or [])
        if not keys:
            raise ValueError("Select at least one join key.")
        result = clean[0]
        for idx, right in enumerate(clean[1:], start=2):
            missing = [k for k in keys if k not in result.columns or k not in right.columns]
            if missing:
                raise ValueError(f"Join key(s) missing in dataset {idx}: {missing}")
            result = result.merge(right, on=keys, how=join_how, suffixes=("", f"__d{idx}"))
        return result
    return clean[0]


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_share = float(numeric.notna().mean())
        rows.append({
            "variable": col,
            "dtype": str(s.dtype),
            "role_guess": infer_role(s),
            "non_missing_n": int(s.notna().sum()),
            "missing_n": int(s.isna().sum()),
            "missing_pct": 100 * float(s.isna().sum()) / n,
            "unique_n": int(s.nunique(dropna=True)),
            "numeric_parse_pct": 100 * numeric_share,
            "example": next((str(v) for v in s.dropna().head(1)), ""),
        })
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Robust univariate IQR/MAD flags without deleting any observation."""
    cols = list(columns) if columns is not None else list(df.select_dtypes(include=np.number).columns)
    rows = []
    for c in cols[:1000]:
        if c not in df:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) < 4 or x.nunique() < 2:
            continue
        q1, median, q3 = x.quantile([.25, .5, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((x < lower) | (x > upper)).sum())
        mad = float(np.median(np.abs(x - median)))
        rows.append({
            "variable": c, "n": len(x), "median": median, "q1": q1, "q3": q3,
            "iqr": iqr, "mad": mad, "lower_fence": lower, "upper_fence": upper,
            "iqr_outliers": count, "outlier_pct": 100 * count / len(x),
        })
    return pd.DataFrame(rows).sort_values(["outlier_pct", "iqr_outliers"], ascending=False) if rows else pd.DataFrame()


def infer_role(s: pd.Series) -> str:
    name = str(s.name).lower()
    unique = s.nunique(dropna=True)
    if any(x in name for x in ("date", "year", "month", "ημερ", "έτος")):
        return "time/date candidate"
    if any(x in name for x in ("id", "code", "κωδ", "a_a")) and unique > 0.8 * max(len(s), 1):
        return "identifier candidate"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if unique == 2 else "numeric"
    if unique <= min(30, max(2, len(s) // 10)):
        return "categorical"
    return "text/identifier"


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_rows = int(df.duplicated().sum())
    constant = int(sum(df[c].nunique(dropna=True) <= 1 for c in df.columns))
    all_missing = int(sum(df[c].isna().all() for c in df.columns))
    mixed = 0
    for c in df.select_dtypes(include="object").columns:
        parsed = pd.to_numeric(df[c], errors="coerce")
        share = parsed.notna().mean()
        if 0.05 < share < 0.95:
            mixed += 1
    return pd.DataFrame([
        {"check": "Rows", "value": len(df), "comment": "Analytical records currently loaded."},
        {"check": "Variables", "value": df.shape[1], "comment": "Columns including source metadata."},
        {"check": "Duplicate rows", "value": duplicate_rows, "comment": "Exact duplicates; review before inferential analysis."},
        {"check": "Constant variables", "value": constant, "comment": "Cannot identify regression effects."},
        {"check": "All-missing variables", "value": all_missing, "comment": "No usable observations."},
        {"check": "Mixed numeric/text variables", "value": mixed, "comment": "Potential coding or decimal-separator issue."},
    ])


def descriptive_statistics(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    cols = list(columns or df.select_dtypes(include=np.number).columns)
    rows = []
    for c in cols:
        if c not in df:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if x.empty:
            continue
        q = x.quantile([.01, .05, .25, .5, .75, .95, .99])
        rows.append({
            "variable": c, "n": len(x), "missing_n": int(df[c].isna().sum()),
            "mean": x.mean(), "std_dev": x.std(ddof=1), "std_error": x.sem(),
            "min": x.min(), "p01": q.loc[.01], "p05": q.loc[.05], "q1": q.loc[.25],
            "median": q.loc[.5], "q3": q.loc[.75], "p95": q.loc[.95], "p99": q.loc[.99],
            "max": x.max(), "iqr": q.loc[.75] - q.loc[.25],
            "skewness": x.skew(), "kurtosis_excess": x.kurt(),
            "cv": x.std(ddof=1) / x.mean() if x.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, columns: Sequence[str], top_n: int = 50) -> pd.DataFrame:
    out = []
    for c in columns:
        if c not in df:
            continue
        counts = df[c].astype("string").fillna("<missing>").value_counts(dropna=False).head(top_n)
        for value, n in counts.items():
            out.append({"variable": c, "category": value, "n": int(n), "pct": 100 * n / max(len(df), 1)})
    return pd.DataFrame(out)


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str], method: str = "pearson") -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in columns if c in df.columns]
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method=method, min_periods=3)
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        pvals.loc[a, a] = 0.0
        for b in cols[i + 1:]:
            z = d[[a, b]].dropna()
            if len(z) < 3 or z[a].nunique() < 2 or z[b].nunique() < 2:
                continue
            if method == "spearman":
                _, p = stats.spearmanr(z[a], z[b])
            elif method == "kendall":
                _, p = stats.kendalltau(z[a], z[b])
            else:
                _, p = stats.pearsonr(z[a], z[b])
            pvals.loc[a, b] = pvals.loc[b, a] = p
    return corr, pvals


def group_tests(df: pd.DataFrame, outcomes: Sequence[str], group: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group not in df:
        return pd.DataFrame()
    for y in outcomes:
        if y not in df:
            continue
        d = df[[y, group]].copy()
        d[y] = pd.to_numeric(d[y], errors="coerce")
        d = d.dropna()
        samples = [g[y].to_numpy() for _, g in d.groupby(group) if len(g) >= 2]
        labels = [str(k) for k, g in d.groupby(group) if len(g) >= 2]
        if len(samples) < 2:
            continue
        if len(samples) == 2:
            t, p = stats.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit")
            u, pu = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            n1, n2 = len(samples[0]), len(samples[1])
            pooled = np.sqrt(((n1 - 1) * np.var(samples[0], ddof=1) + (n2 - 1) * np.var(samples[1], ddof=1)) / max(n1 + n2 - 2, 1))
            cohens_d = (np.mean(samples[0]) - np.mean(samples[1])) / pooled if pooled > 0 else np.nan
            correction = 1 - 3 / max(4 * (n1 + n2) - 9, 1)
            hedges_g = correction * cohens_d if np.isfinite(cohens_d) else np.nan
            rank_biserial = 1 - 2 * u / (n1 * n2)
            rows.extend([
                {"outcome": y, "group": group, "levels": " | ".join(labels), "test": "Welch t-test", "statistic": t, "p_value": p, "effect_metric": "Hedges g", "effect_size": hedges_g},
                {"outcome": y, "group": group, "levels": " | ".join(labels), "test": "Mann–Whitney U", "statistic": u, "p_value": pu, "effect_metric": "Rank-biserial correlation", "effect_size": rank_biserial},
            ])
        else:
            f, p = stats.f_oneway(*samples)
            h, ph = stats.kruskal(*samples)
            all_values = np.concatenate(samples)
            grand = np.mean(all_values)
            ss_between = sum(len(s) * (np.mean(s) - grand) ** 2 for s in samples)
            ss_total = np.sum((all_values - grand) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
            epsilon_sq = max(0.0, (h - len(samples) + 1) / max(len(all_values) - len(samples), 1))
            rows.extend([
                {"outcome": y, "group": group, "levels": len(samples), "test": "One-way ANOVA", "statistic": f, "p_value": p, "effect_metric": "Eta squared", "effect_size": eta_sq},
                {"outcome": y, "group": group, "levels": len(samples), "test": "Kruskal–Wallis", "statistic": h, "p_value": ph, "effect_metric": "Epsilon squared", "effect_size": epsilon_sq},
            ])
        lev, pl = stats.levene(*samples, center="median")
        rows.append({"outcome": y, "group": group, "levels": len(samples), "test": "Levene (median)", "statistic": lev, "p_value": pl, "effect_metric": "Assumption diagnostic", "effect_size": np.nan})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["significant_5pct"] = result["p_value"] < .05
    return result


def chi_square_tests(df: pd.DataFrame, variables: Sequence[str], group: str) -> pd.DataFrame:
    rows = []
    for variable in variables:
        if variable not in df or group not in df or variable == group:
            continue
        tab = pd.crosstab(df[variable], df[group])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(tab)
        n = tab.to_numpy().sum()
        denom = max(min(tab.shape) - 1, 1)
        cramers_v = math.sqrt((chi2 / n) / denom) if n else np.nan
        rows.append({"variable": variable, "group": group, "chi_square": chi2, "dof": dof,
                     "p_value": p, "cramers_v": cramers_v, "min_expected": expected.min()})
    return pd.DataFrame(rows)


def normality_tests(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        x = pd.to_numeric(df[c], errors="coerce").dropna() if c in df else pd.Series(dtype=float)
        if len(x) < 3:
            continue
        sample = x.sample(min(len(x), 5000), random_state=42) if len(x) > 5000 else x
        sh_stat, sh_p = stats.shapiro(sample)
        da_stat, da_p = stats.normaltest(x) if len(x) >= 8 else (np.nan, np.nan)
        ad = stats.anderson(x, dist="norm")
        rows.extend([
            {"variable": c, "test": "Shapiro–Wilk", "n_used": len(sample), "statistic": sh_stat, "p_value": sh_p},
            {"variable": c, "test": "D'Agostino K²", "n_used": len(x), "statistic": da_stat, "p_value": da_p},
            {"variable": c, "test": "Anderson–Darling", "n_used": len(x), "statistic": ad.statistic, "p_value": np.nan},
        ])
    return pd.DataFrame(rows)


def _encode_design(df: pd.DataFrame, x_vars: Sequence[str], categorical: Sequence[str], add_constant: bool = True) -> pd.DataFrame:
    x = df[list(dict.fromkeys(x_vars))].copy()
    cats = [c for c in categorical if c in x]
    for c in x.columns:
        if c not in cats:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = pd.get_dummies(x, columns=cats, drop_first=True, dtype=float)
    x = x.loc[:, x.nunique(dropna=True) > 1]
    if add_constant:
        x = sm.add_constant(x, has_constant="add")
    return x.astype(float)


@dataclass
class ModelOutput:
    coefficients: pd.DataFrame
    fit: pd.DataFrame
    diagnostics: pd.DataFrame
    predictions: pd.DataFrame
    interpretation: list[str]
    raw_result: Any | None = None


def fit_detailed_model(
    df: pd.DataFrame,
    y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    estimator: str = "OLS",
    covariance: str = "HC3",
    cluster: str | None = None,
    weights: str | None = None,
    quantile: float = .5,
) -> ModelOutput:
    required = list(dict.fromkeys([y, *x_vars, *categorical, *([cluster] if cluster else []), *([weights] if weights else [])]))
    d = df[[c for c in required if c in df]].copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    X = _encode_design(d, x_vars, categorical)
    joined = pd.concat([d[[y]], X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < max(10, X.shape[1] + 2):
        raise ValueError("Too few complete observations for the selected model.")
    yy = joined[y].astype(float)
    XX = joined.drop(columns=y).astype(float)
    groups = d.loc[joined.index, cluster] if cluster and cluster in d else None
    fit_kw: dict[str, Any] = {}
    cov_label = covariance
    if groups is not None and groups.nunique() > 1:
        fit_kw = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
        cov_label = f"clustered by {cluster}"
    elif covariance in {"HC0", "HC1", "HC2", "HC3", "HAC"}:
        fit_kw = {"cov_type": covariance}
        if covariance == "HAC":
            fit_kw["cov_kwds"] = {"maxlags": 1}

    if estimator == "OLS":
        result = sm.OLS(yy, XX).fit(**fit_kw)
    elif estimator == "WLS":
        if not weights or weights not in d:
            raise ValueError("WLS requires a positive weights variable.")
        w = pd.to_numeric(d.loc[joined.index, weights], errors="coerce").clip(lower=np.finfo(float).eps)
        result = sm.WLS(yy, XX, weights=w).fit(**fit_kw)
    elif estimator == "Logit":
        result = sm.GLM(yy, XX, family=sm.families.Binomial()).fit(**fit_kw)
    elif estimator == "Probit":
        result = sm.Probit(yy, XX).fit(disp=False, **fit_kw)
    elif estimator == "Poisson":
        result = sm.GLM(yy, XX, family=sm.families.Poisson()).fit(**fit_kw)
    elif estimator == "Negative binomial":
        result = sm.GLM(yy, XX, family=sm.families.NegativeBinomial(alpha=1.0)).fit(**fit_kw)
    elif estimator == "Fractional logit":
        if (yy.lt(0) | yy.gt(1)).any():
            raise ValueError("Fractional logit requires an outcome in [0, 1].")
        result = sm.GLM(yy, XX, family=sm.families.Binomial()).fit(**fit_kw)
    elif estimator == "Quantile regression":
        result = sm.QuantReg(yy, XX).fit(q=quantile)
        cov_label = "quantile-regression covariance"
    elif estimator == "Robust Huber":
        result = sm.RLM(yy, XX, M=sm.robust.norms.HuberT()).fit()
        cov_label = "Huber M-estimation"
    elif estimator == "Gamma log-link":
        if (yy <= 0).any():
            raise ValueError("Gamma regression requires a strictly positive outcome.")
        result = sm.GLM(yy, XX, family=sm.families.Gamma(link=sm.families.links.Log())).fit(**fit_kw)
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")

    ci = result.conf_int()
    coef = pd.DataFrame({
        "term": result.params.index, "coefficient": result.params.values,
        "std_error": np.asarray(result.bse), "statistic": np.asarray(result.tvalues),
        "p_value": np.asarray(result.pvalues), "ci_95_low": ci.iloc[:, 0].values,
        "ci_95_high": ci.iloc[:, 1].values,
    })
    coef["significance"] = pd.cut(coef.p_value, [-np.inf, .001, .01, .05, .1, np.inf], labels=["***", "**", "*", ".", ""])
    if estimator in {"Logit", "Poisson", "Negative binomial", "Fractional logit", "Gamma log-link"}:
        coef["exp_coefficient"] = np.exp(coef.coefficient)
        coef["exp_ci_95_low"] = np.exp(coef.ci_95_low)
        coef["exp_ci_95_high"] = np.exp(coef.ci_95_high)
        coef["effect_scale"] = "odds ratio" if estimator in {"Logit", "Fractional logit"} else "multiplicative mean ratio"
    bic_value = safe_result_metric(result, "bic_llf")
    if not np.isfinite(bic_value):
        bic_value = safe_result_metric(result, "bic")
    fit = pd.DataFrame([{
        "estimator": estimator, "covariance": cov_label, "n": int(result.nobs),
        "parameters": int(len(result.params)), "log_likelihood": safe_result_metric(result, "llf"),
        "aic": safe_result_metric(result, "aic"), "bic": bic_value,
        "r_squared": safe_result_metric(result, "rsquared"),
        "adjusted_r_squared": safe_result_metric(result, "rsquared_adj"),
        "pseudo_r_squared": safe_result_metric(result, "prsquared"),
        "deviance": safe_result_metric(result, "deviance"),
    }])
    pred = np.asarray(result.predict(XX))
    predictions = pd.DataFrame({"row_index": joined.index, "observed": yy, "fitted": pred, "residual": yy - pred})
    diagnostics = regression_diagnostics(result, XX, yy) if estimator in {"OLS", "WLS"} else glm_diagnostics(result, yy, pred)
    interpretation = interpret_model(fit.iloc[0].to_dict(), coef, diagnostics, y)
    return ModelOutput(coef, fit, diagnostics, predictions, interpretation, result)


def regression_diagnostics(result: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    resid = np.asarray(result.resid)
    rows: list[dict[str, Any]] = []
    jb, jbp, skew, kurt = jarque_bera(resid)
    rows.append({"diagnostic": "Jarque–Bera normality", "statistic": jb, "p_value": jbp, "detail": f"skew={skew:.3f}; kurtosis={kurt:.3f}"})
    if len(resid) >= 8:
        omni, omnip = omni_normtest(resid)
        rows.append({"diagnostic": "Omnibus normality", "statistic": omni, "p_value": omnip, "detail": "Residual normality test"})
    lm, lmp, fval, fp = het_breuschpagan(resid, X)
    rows.append({"diagnostic": "Breusch–Pagan heteroskedasticity", "statistic": lm, "p_value": lmp, "detail": f"F p={fp:.4g}"})
    if X.shape[1] <= 80:
        try:
            lm, lmp, fval, fp = het_white(resid, X)
            rows.append({"diagnostic": "White heteroskedasticity", "statistic": lm, "p_value": lmp, "detail": f"F p={fp:.4g}"})
        except Exception:
            pass
    rows.append({"diagnostic": "Durbin–Watson", "statistic": durbin_watson(resid), "p_value": np.nan, "detail": "Approximately 2 indicates little first-order autocorrelation."})
    try:
        reset = linear_reset(result, power=2, use_f=True)
        rows.append({"diagnostic": "Ramsey RESET", "statistic": float(reset.fvalue), "p_value": float(reset.pvalue), "detail": "Functional-form check"})
    except Exception:
        pass
    influence = OLSInfluence(result)
    cooks = influence.cooks_distance[0]
    rows.append({"diagnostic": "Influential observations", "statistic": int(np.sum(cooks > 4 / max(len(cooks), 1))), "p_value": np.nan, "detail": "Count with Cook's D > 4/n"})
    condition = np.linalg.cond(np.asarray(X)) if X.shape[1] else np.nan
    rows.append({"diagnostic": "Condition number", "statistic": condition, "p_value": np.nan, "detail": "Large values indicate scaling or collinearity concerns."})
    return pd.DataFrame(rows)


def glm_diagnostics(result: Any, y: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    rmse = float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))
    mae = float(np.mean(np.abs(np.asarray(y) - pred)))
    pearson = getattr(result, "pearson_chi2", np.nan)
    df_resid = getattr(result, "df_resid", np.nan)
    dispersion = pearson / df_resid if np.isfinite(pearson) and df_resid else np.nan
    return pd.DataFrame([
        {"diagnostic": "RMSE", "statistic": rmse, "p_value": np.nan, "detail": "Prediction error in outcome units."},
        {"diagnostic": "MAE", "statistic": mae, "p_value": np.nan, "detail": "Mean absolute prediction error."},
        {"diagnostic": "Pearson dispersion", "statistic": dispersion, "p_value": np.nan, "detail": "Values well above 1 indicate overdispersion."},
    ])


def vif_table(df: pd.DataFrame, x_vars: Sequence[str], categorical: Sequence[str] = ()) -> pd.DataFrame:
    X = _encode_design(df, x_vars, categorical, add_constant=False).replace([np.inf, -np.inf], np.nan).dropna()
    if X.shape[1] < 2:
        return pd.DataFrame({"message": ["At least two non-constant regressors are required."]})
    if X.shape[1] > 250:
        return pd.DataFrame({"message": ["VIF is intentionally limited to 250 encoded regressors; use correlation screening or regularisation first."]})
    rows = []
    for i, c in enumerate(X.columns):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                value = variance_inflation_factor(X.values, i)
            rows.append({"variable": c, "VIF": value})
        except Exception as exc:
            rows.append({"variable": c, "VIF": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def matrix_ols_many_outcomes(
    df: pd.DataFrame,
    y_vars: Sequence[str],
    x_vars: Sequence[str],
    missing: str = "median imputation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fast OLS for up to 1,000 outcomes × 1,000 predictors.

    Uses one SVD/pseudoinverse and vectorised coefficient/standard-error
    calculations. This is the scalable screening engine; detailed robust
    diagnostics remain available in ``fit_detailed_model``.
    """
    ys = list(dict.fromkeys(y_vars))[:MAX_DEPENDENT]
    xs = list(dict.fromkeys(x_vars))[:MAX_INDEPENDENT]
    if not ys or not xs:
        raise ValueError("Select at least one dependent and one independent variable.")
    raw = df[ys + xs].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if missing == "complete cases":
        raw = raw.dropna()
    else:
        raw = raw.fillna(raw.median(numeric_only=True)).dropna(axis=1, how="all")
        ys = [c for c in ys if c in raw]
        xs = [c for c in xs if c in raw]
    if len(raw) < 5 or not ys or not xs:
        raise ValueError("Insufficient usable observations after missing-data handling.")
    X = np.column_stack([np.ones(len(raw)), raw[xs].to_numpy(float)])
    Y = raw[ys].to_numpy(float)
    rank = int(np.linalg.matrix_rank(X))
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    fitted = X @ beta
    resid = Y - fitted
    dof = max(len(raw) - rank, 1)
    sigma2 = np.sum(resid ** 2, axis=0) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv)[:, None] * sigma2[None, :], 0, np.inf))
    tvals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    terms = ["const", *xs]
    cells = len(terms) * len(ys)
    coef = pd.DataFrame({
        "outcome": np.repeat(np.asarray(ys, dtype=object), len(terms)),
        "term": np.tile(np.asarray(terms, dtype=object), len(ys)),
        "coefficient": beta.T.reshape(-1),
        "std_error": se.T.reshape(-1),
        "t_statistic": tvals.T.reshape(-1),
        "p_value": pvals.T.reshape(-1),
    })
    sst = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    sse = np.sum(resid ** 2, axis=0)
    r2 = 1 - np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 0)
    fit = pd.DataFrame({"outcome": ys, "n": len(raw), "predictors": len(xs), "rank": rank, "degrees_freedom": dof, "r_squared": r2, "rmse": np.sqrt(sse / len(raw))})
    fit["coefficient_cells"] = cells
    return coef, fit


def monte_carlo_ols(
    df: pd.DataFrame,
    y: str,
    x_vars: Sequence[str],
    simulations: int = 2_000,
    method: str = "Wild bootstrap",
    confidence: float = .95,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the finite-sample uncertainty of an OLS specification.

    Three data-generating mechanisms are supported. Residual bootstrap samples
    centred residuals with replacement; wild bootstrap multiplies residuals by
    Rademacher weights and is preferable under unknown heteroskedasticity;
    parametric normal simulation uses the estimated residual standard deviation.
    The linear algebra is chunked so several thousand replications remain usable
    on Streamlit Community Cloud.
    """
    xs = list(dict.fromkeys(c for c in x_vars if c in df and c != y))
    if not xs:
        raise ValueError("Select at least one independent variable.")
    if len(xs) > 100:
        raise ValueError("Monte Carlo OLS is limited to 100 predictors per run; use the 1,000 x 1,000 engine for screening first.")
    simulations = int(np.clip(simulations, 100, 20_000))
    confidence = float(np.clip(confidence, .80, .999))
    data = df[[y, *xs]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(data[xs], has_constant="add").astype(float)
    yy = data[y].astype(float).to_numpy()
    if len(data) < X.shape[1] + 10:
        raise ValueError("Too few complete observations for stable simulation relative to the number of predictors.")
    matrix = X.to_numpy(float)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise ValueError("The OLS design matrix is rank deficient. Remove duplicated or collinear predictors before simulation.")
    pinv = np.linalg.pinv(matrix)
    beta = pinv @ yy
    fitted = matrix @ beta
    residual = yy - fitted
    residual = residual - residual.mean()
    sigma = float(np.sqrt(np.sum(residual ** 2) / max(len(data) - rank, 1)))
    rng = np.random.default_rng(int(seed))
    draws = np.empty((simulations, matrix.shape[1]), dtype=float)
    chunk = max(25, min(500, 2_000_000 // max(len(data), 1)))
    for start in range(0, simulations, chunk):
        width = min(chunk, simulations - start)
        if method == "Residual bootstrap":
            errors = rng.choice(residual, size=(len(data), width), replace=True)
        elif method == "Parametric normal":
            errors = rng.normal(0.0, sigma, size=(len(data), width))
        else:
            errors = residual[:, None] * rng.choice(np.array([-1.0, 1.0]), size=(len(data), width))
        draws[start:start + width] = (pinv @ (fitted[:, None] + errors)).T
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1 - alpha], axis=0)
    prob_positive = np.mean(draws > 0, axis=0)
    terms = list(X.columns)
    summary = pd.DataFrame({
        "term": terms,
        "ols_estimate": beta,
        "simulation_mean": draws.mean(axis=0),
        "simulation_median": np.median(draws, axis=0),
        "simulation_bias": draws.mean(axis=0) - beta,
        "monte_carlo_se": draws.std(axis=0, ddof=1),
        f"ci_{confidence:.1%}_low": lower,
        f"ci_{confidence:.1%}_high": upper,
        "probability_positive": prob_positive,
        "two_sided_simulation_p": np.minimum(1.0, 2 * np.minimum(prob_positive, 1 - prob_positive)),
    })
    draw_table = pd.DataFrame(draws, columns=terms)
    draw_table.insert(0, "simulation", np.arange(1, simulations + 1))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1 - float(np.sum(residual ** 2)) / sst if sst else np.nan
    fit = pd.DataFrame([{
        "method": method, "simulations": simulations, "seed": int(seed),
        "confidence_level": confidence, "complete_observations": len(data),
        "predictors": len(xs), "design_rank": rank, "base_ols_r_squared": r2,
        "residual_sigma": sigma,
    }])
    return summary, draw_table, fit


def monte_carlo_portfolio(
    df: pd.DataFrame,
    cost: str,
    benefit: str,
    budget: float,
    project_id: str | None = None,
    simulations: int = 1_000,
    cost_cv: float = .10,
    benefit_cv: float = .20,
    correlation: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stochastic budget-constrained portfolio ranking.

    Each replication draws correlated lognormal cost and benefit multipliers,
    ranks projects by simulated benefit/cost and greedily fills the budget.
    Selection frequency is a transparent robustness measure, not a claim that
    the greedy solution is globally optimal for every policy constraint.
    """
    if cost == benefit:
        raise ValueError("Cost and benefit must be different variables.")
    cols = [cost, benefit] + ([project_id] if project_id and project_id in df else [])
    data = df[cols].copy()
    data[cost] = pd.to_numeric(data[cost], errors="coerce")
    data[benefit] = pd.to_numeric(data[benefit], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[cost, benefit])
    data = data[(data[cost] > 0) & (data[benefit] >= 0)].copy()
    if data.empty:
        raise ValueError("No observations have a positive cost and non-negative benefit.")
    if len(data) > 5_000:
        raise ValueError("Portfolio simulation is limited to 5,000 eligible projects per run.")
    budget = float(budget)
    if budget <= 0:
        raise ValueError("The available budget must be positive.")
    simulations = int(np.clip(simulations, 100, 10_000))
    cost_cv = float(np.clip(cost_cv, 0, 2.0))
    benefit_cv = float(np.clip(benefit_cv, 0, 2.0))
    correlation = float(np.clip(correlation, -.95, .95))
    base_cost = data[cost].to_numpy(float)
    base_benefit = data[benefit].to_numpy(float)
    n = len(data)
    selected = np.zeros(n, dtype=int)
    selected_benefit = np.zeros(n, dtype=float)
    selected_cost = np.zeros(n, dtype=float)
    totals = np.empty((simulations, 3), dtype=float)
    sigma_c = np.sqrt(np.log1p(cost_cv ** 2))
    sigma_b = np.sqrt(np.log1p(benefit_cv ** 2))
    mu_c, mu_b = -.5 * sigma_c ** 2, -.5 * sigma_b ** 2
    rng = np.random.default_rng(int(seed))
    for s in range(simulations):
        z1 = rng.normal(size=n)
        z2 = correlation * z1 + np.sqrt(1 - correlation ** 2) * rng.normal(size=n)
        sim_cost = base_cost * np.exp(mu_c + sigma_c * z1)
        sim_benefit = base_benefit * np.exp(mu_b + sigma_b * z2)
        ratio = np.divide(sim_benefit, sim_cost, out=np.zeros_like(sim_benefit), where=sim_cost > 0)
        order = np.argsort(-ratio)
        cumulative = np.cumsum(sim_cost[order])
        chosen = order[cumulative <= budget]
        if not len(chosen) and sim_cost[order[0]] <= budget:
            chosen = order[:1]
        selected[chosen] += 1
        selected_benefit[chosen] += sim_benefit[chosen]
        selected_cost[chosen] += sim_cost[chosen]
        totals[s] = (sim_cost[chosen].sum(), sim_benefit[chosen].sum(), len(chosen))
    ids = data[project_id].astype(str).to_numpy() if project_id and project_id in data else data.index.astype(str).to_numpy()
    project_table = pd.DataFrame({
        "project_id": ids,
        "base_cost": base_cost,
        "base_benefit": base_benefit,
        "base_benefit_cost_ratio": np.divide(base_benefit, base_cost),
        "selection_probability": selected / simulations,
        "times_selected": selected,
        "mean_cost_when_selected": np.divide(selected_cost, selected, out=np.full(n, np.nan), where=selected > 0),
        "mean_benefit_when_selected": np.divide(selected_benefit, selected, out=np.full(n, np.nan), where=selected > 0),
    }).sort_values(["selection_probability", "base_benefit_cost_ratio"], ascending=False)
    simulation_table = pd.DataFrame(totals, columns=["portfolio_cost", "portfolio_benefit", "projects_selected"])
    simulation_table.insert(0, "simulation", np.arange(1, simulations + 1))
    summary = pd.DataFrame([{
        "eligible_projects": n, "simulations": simulations, "budget": budget,
        "cost_cv": cost_cv, "benefit_cv": benefit_cv, "shock_correlation": correlation,
        "mean_portfolio_cost": simulation_table.portfolio_cost.mean(),
        "mean_portfolio_benefit": simulation_table.portfolio_benefit.mean(),
        "p05_portfolio_benefit": simulation_table.portfolio_benefit.quantile(.05),
        "median_portfolio_benefit": simulation_table.portfolio_benefit.median(),
        "p95_portfolio_benefit": simulation_table.portfolio_benefit.quantile(.95),
        "mean_projects_selected": simulation_table.projects_selected.mean(),
        "seed": int(seed),
    }])
    return summary, project_table, simulation_table


def p_adjust(pvalues: Sequence[float], method: str = "Benjamini–Hochberg") -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    vals = p[mask]
    m = len(vals)
    if m == 0:
        return out
    if method == "Bonferroni":
        adj = np.minimum(vals * m, 1)
    else:
        order = np.argsort(vals)
        ranked = vals[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(ranked, 1)
    out[mask] = adj
    return out


def interpret_model(fit: Mapping[str, Any], coef: pd.DataFrame, diagnostics: pd.DataFrame, outcome: str) -> list[str]:
    comments = [f"The model for “{outcome}” uses {int(fit.get('n', 0)):,} complete observations and {int(fit.get('parameters', 0))} estimated parameters."]
    r2 = fit.get("adjusted_r_squared")
    if not np.isfinite(r2 if r2 is not None else np.nan):
        r2 = fit.get("r_squared")
    if np.isfinite(r2 if r2 is not None else np.nan):
        comments.append(f"Adjusted/model R² is {float(r2):.3f}; this is descriptive fit, not evidence of causality.")
    sig = coef[(coef.term != "const") & (coef.p_value < .05)].sort_values("p_value")
    if sig.empty:
        comments.append("No non-constant coefficient is statistically distinguishable from zero at the 5% level in this specification.")
    else:
        top = sig.head(5)
        bits = [f"{r.term} ({'+' if r.coefficient > 0 else '−'}, p={r.p_value:.3g})" for r in top.itertuples()]
        comments.append("The strongest conditional associations are: " + "; ".join(bits) + ".")
    for row in diagnostics.itertuples():
        if getattr(row, "p_value", np.nan) < .05 and any(k in row.diagnostic for k in ("heteroskedasticity", "RESET", "normality")):
            comments.append(f"{row.diagnostic} rejects its null at 5%; use robust inference and inspect specification/transformations.")
    comments.append("Coefficient signs and p-values must be interpreted against the variable definitions, sampling process, multiple-testing burden and identification assumptions.")
    return comments


def time_series_tests(df: pd.DataFrame, variables: Sequence[str], time_col: str) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    d = df.sort_values(time_col)
    for c in variables:
        x = pd.to_numeric(d[c], errors="coerce").dropna() if c in d else pd.Series(dtype=float)
        if len(x) < 12 or x.nunique() < 3:
            continue
        try:
            adf = adfuller(x, autolag="AIC")
            rows.append({"variable": c, "test": "ADF unit root", "statistic": adf[0], "p_value": adf[1], "lags": adf[2], "n": adf[3]})
        except Exception:
            pass
        try:
            kp = kpss(x, regression="c", nlags="auto")
            rows.append({"variable": c, "test": "KPSS stationarity", "statistic": kp[0], "p_value": kp[1], "lags": kp[2], "n": len(x)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def pca_table(df: pd.DataFrame, columns: Sequence[str], n_components: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in columns if c in df]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    n_components = min(n_components, x.shape[0], x.shape[1])
    model = PCA(n_components=n_components, random_state=42).fit(x)
    loadings = pd.DataFrame(model.components_.T, index=cols, columns=[f"PC{i+1}" for i in range(n_components)]).reset_index(names="variable")
    variance = pd.DataFrame({"component": [f"PC{i+1}" for i in range(n_components)], "explained_variance_ratio": model.explained_variance_ratio_, "cumulative": np.cumsum(model.explained_variance_ratio_)})
    return loadings, variance


def cluster_table(df: pd.DataFrame, columns: Sequence[str], clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in columns if c in df]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    matrix = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x))
    clusters = min(max(2, clusters), len(df))
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=20).fit_predict(matrix)
    assignments = pd.DataFrame({"row_index": df.index, "cluster": labels})
    prof = df[cols].apply(pd.to_numeric, errors="coerce").assign(cluster=labels).groupby("cluster").mean().reset_index()
    return assignments, prof


def regularised_regression(
    df: pd.DataFrame,
    y: str,
    x_vars: Sequence[str],
    method: str = "Ridge",
    alpha: float = 1.0,
    l1_ratio: float = .5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardised Ridge/Lasso/Elastic Net with a deterministic hold-out."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in x_vars if c in df]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    Y = pd.to_numeric(df[y], errors="coerce")
    keep = Y.notna()
    X, Y = X.loc[keep], Y.loc[keep]
    if len(Y) < 20:
        raise ValueError("At least 20 observations are required for hold-out regularisation.")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    if method == "Lasso": model = Lasso(alpha=alpha, max_iter=20_000, random_state=42)
    elif method == "Elastic Net": model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000, random_state=42)
    else: model = Ridge(alpha=alpha, random_state=42)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    coefficients = pd.DataFrame({"variable": cols, "standardised_coefficient": model.coef_}).sort_values("standardised_coefficient", key=np.abs, ascending=False)
    fit = pd.DataFrame([{"method": method, "alpha": alpha, "l1_ratio": l1_ratio if method == "Elastic Net" else np.nan, "train_n": len(y_train), "test_n": len(y_test), "test_r_squared": r2_score(y_test, pred), "test_rmse": np.sqrt(mean_squared_error(y_test, pred)), "test_mae": mean_absolute_error(y_test, pred)}])
    return coefficients, fit


def instrumental_variables_2sls(
    df: pd.DataFrame,
    y: str,
    endogenous: str,
    instruments: Sequence[str],
    exogenous: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stage least squares using excluded instruments plus exogenous controls."""
    from statsmodels.sandbox.regression.gmm import IV2SLS
    cols = list(dict.fromkeys([y, endogenous, *instruments, *exogenous]))
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < len(cols) + 10:
        raise ValueError("Insufficient complete observations for IV/2SLS.")
    X = sm.add_constant(d[[endogenous, *exogenous]], has_constant="add")
    Z = sm.add_constant(d[[*instruments, *exogenous]], has_constant="add")
    result = IV2SLS(d[y], X, Z).fit()
    ci = result.conf_int()
    coef = pd.DataFrame({"term": result.params.index, "coefficient": result.params.values, "std_error": result.bse, "t_statistic": result.tvalues, "p_value": result.pvalues, "ci_95_low": ci.iloc[:, 0], "ci_95_high": ci.iloc[:, 1]})
    # First-stage relevance diagnostic for the endogenous regressor.
    first = sm.OLS(d[endogenous], Z).fit()
    excluded = [c for c in instruments if c in first.params.index]
    restriction = np.zeros((len(excluded), len(first.params)))
    for row, name in enumerate(excluded): restriction[row, list(first.params.index).index(name)] = 1
    f_test = first.f_test(restriction) if excluded else None
    fit = pd.DataFrame([{"n": len(d), "endogenous": endogenous, "excluded_instruments": "; ".join(instruments), "first_stage_r_squared": first.rsquared, "excluded_instrument_F": float(f_test.fvalue) if f_test is not None else np.nan, "excluded_instrument_p": float(f_test.pvalue) if f_test is not None else np.nan, "second_stage_r_squared": result.rsquared}])
    return coef, fit


def difference_in_differences(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    post: str,
    controls: Sequence[str] = (),
    cluster: str | None = None,
) -> ModelOutput:
    d = df.copy()
    d["__treatment__"] = pd.to_numeric(d[treatment], errors="coerce")
    d["__post__"] = pd.to_numeric(d[post], errors="coerce")
    d["__did__"] = d["__treatment__"] * d["__post__"]
    out = fit_detailed_model(d, y, ["__treatment__", "__post__", "__did__", *controls], estimator="OLS", covariance="HC3", cluster=cluster)
    out.interpretation.insert(0, "The coefficient on __did__ is the difference-in-differences estimate. A causal interpretation requires parallel pre-trends, no anticipatory response and no contemporaneous differential shock.")
    return out


def granger_table(df: pd.DataFrame, cause: str, effect: str, time_col: str, max_lag: int = 4) -> pd.DataFrame:
    from statsmodels.tsa.stattools import grangercausalitytests
    d = df.sort_values(time_col)[[effect, cause]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < max(20, 5 * max_lag):
        raise ValueError("Too few ordered observations for the requested Granger lags.")
    tests = grangercausalitytests(d, maxlag=max_lag, verbose=False)
    rows = []
    for lag, result in tests.items():
        stat, p, df_denom, df_num = result[0]["ssr_ftest"]
        rows.append({"cause": cause, "effect": effect, "lag": lag, "F_statistic": stat, "p_value": p, "df_denom": df_denom, "df_num": df_num})
    return pd.DataFrame(rows)


def arima_forecast(df: pd.DataFrame, variable: str, time_col: str, order: tuple[int, int, int], steps: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    from statsmodels.tsa.arima.model import ARIMA
    d = df.sort_values(time_col)[[time_col, variable]].copy()
    d[variable] = pd.to_numeric(d[variable], errors="coerce")
    d = d.dropna()
    if len(d) < max(20, sum(order) * 4 + 5):
        raise ValueError("Too few observations for this ARIMA order.")
    model = ARIMA(d[variable].to_numpy(), order=order).fit()
    forecast = model.get_forecast(steps=steps)
    ci = np.asarray(forecast.conf_int())
    table = pd.DataFrame({"forecast_step": np.arange(1, steps + 1), "forecast": forecast.predicted_mean, "ci_95_low": ci[:, 0], "ci_95_high": ci[:, 1]})
    fit = pd.DataFrame([{"variable": variable, "order": str(order), "n": len(d), "aic": model.aic, "bic": model.bic, "log_likelihood": model.llf}])
    return table, fit


def cronbach_alpha(df: pd.DataFrame, items: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[list(items)].apply(pd.to_numeric, errors="coerce").dropna()
    k = d.shape[1]
    if k < 2 or len(d) < 5:
        raise ValueError("At least two items and five complete cases are required.")
    total = d.sum(axis=1)
    variance_sum = d.var(ddof=1).sum()
    alpha = k / (k - 1) * (1 - variance_sum / total.var(ddof=1))
    rows = []
    for item in d.columns:
        rest = d.drop(columns=item)
        rest_total = rest.sum(axis=1)
        alpha_deleted = (k - 1) / (k - 2) * (1 - rest.var(ddof=1).sum() / rest_total.var(ddof=1)) if k > 2 else np.nan
        rows.append({"item": item, "item_total_correlation": d[item].corr(total - d[item]), "alpha_if_deleted": alpha_deleted})
    summary = pd.DataFrame([{"items": k, "complete_cases": len(d), "cronbach_alpha": alpha}])
    return summary, pd.DataFrame(rows)


def to_excel_bytes(tables: Mapping[str, pd.DataFrame]) -> bytes:
    export_tables = dict(tables)
    if not any("output guide" in str(name).casefold() for name in export_tables):
        from output_guidance import build_output_guide
        export_tables["Output guide"] = build_output_guide(export_tables)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        used: set[str] = set()
        for raw_name, table in export_tables.items():
            name = re.sub(r"[\\/*?:\[\]]", "_", raw_name)[:31] or "Table"
            base = name
            counter = 2
            while name in used:
                suffix = f"_{counter}"
                name = base[:31-len(suffix)] + suffix
                counter += 1
            used.add(name)
            table.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()


def serialisable_summary(df: pd.DataFrame) -> str:
    payload = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(exclude=np.number).columns),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
