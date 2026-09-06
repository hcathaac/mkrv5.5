"""Generalised confirmatory, compositional, ranking and rare-method analytics.

Makryvelios v5.9.1 additive module.

Design goals
------------
* Generic methods, never paper-specific hard-coding.
* No external LLM required for numerical estimation.
* No new runtime dependencies beyond the existing Makryvelios stack
  (NumPy, pandas, SciPy, statsmodels and scikit-learn).
* Explicit diagnostics and conservative naming for approximate procedures.
* Reproducible randomisation via user-controlled seeds.

The functions in this file are intentionally independent of Streamlit so they
can be unit-tested, scripted and reused by Agentic/Research Chair workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import digamma, gammaln, logsumexp
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_ind, ttost_paired
from statsmodels.tools.numdiff import approx_hess


EPS = np.finfo(float).eps


@dataclass
class AnalysisResult:
    """Common result carrier for the confirmatory laboratory."""

    tables: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame
    settings: dict[str, Any]
    warnings: list[str]
    raw_result: Any | None = None


def _as_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns))
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    return df[columns].apply(pd.to_numeric, errors="coerce")


def _design_matrix(
    df: pd.DataFrame,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    *,
    add_constant: bool = True,
    reference_levels: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a numeric design matrix with optional explicit reference categories.

    ``reference_levels`` is deliberately generic and only affects predictors
    listed in ``categorical``.  When supplied, the requested level is placed
    first before dummy encoding, making the reference category deterministic
    across computers, pandas versions and future datasets.
    """
    x_vars = list(dict.fromkeys(x_vars))
    if not x_vars:
        X = pd.DataFrame(index=df.index)
    else:
        missing = [c for c in x_vars if c not in df.columns]
        if missing:
            raise ValueError("Missing predictors: " + ", ".join(missing))
        X = df[x_vars].copy()
    categorical = [c for c in categorical if c in X.columns]
    reference_levels = dict(reference_levels or {})
    for c in X.columns:
        if c not in categorical:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        observed = [v for v in pd.Series(X[c]).dropna().unique().tolist()]
        if not observed:
            continue
        ref = reference_levels.get(c, None)
        if ref is not None:
            if ref not in observed:
                # Allow a string-equivalent value from JSON/UI recipes.
                matches = [v for v in observed if str(v) == str(ref)]
                if not matches:
                    raise ValueError(f"Reference level {ref!r} is not observed in categorical predictor {c!r}.")
                ref = matches[0]
            ordered = [ref] + sorted([v for v in observed if v != ref], key=lambda v: str(v))
            X[c] = pd.Categorical(X[c], categories=ordered, ordered=False)
    if categorical:
        X = pd.get_dummies(X, columns=categorical, drop_first=True, dtype=float)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    return X.astype(float)


def _coef_table(result: Any, *, exponentiate: bool = False, effect_name: str = "ratio") -> pd.DataFrame:
    params = pd.Series(np.asarray(result.params), index=getattr(result.params, "index", None))
    if params.index is None or isinstance(params.index, pd.RangeIndex):
        names = getattr(result.model, "exog_names", [f"x{i}" for i in range(len(params))])
        params.index = names
    bse = np.asarray(result.bse)
    pvals = np.asarray(result.pvalues)
    ci = np.asarray(result.conf_int())
    out = pd.DataFrame({
        "term": params.index.astype(str),
        "coefficient": params.to_numpy(float),
        "std_error": bse,
        "p_value": pvals,
        "ci_95_low": ci[:, 0],
        "ci_95_high": ci[:, 1],
    })
    if exponentiate:
        out[effect_name] = np.exp(out["coefficient"])
        out[f"{effect_name}_ci_95_low"] = np.exp(out["ci_95_low"])
        out[f"{effect_name}_ci_95_high"] = np.exp(out["ci_95_high"])
    return out


# ---------------------------------------------------------------------------
# 1) Rare-event / separation-safe binary regression: Firth logistic
# ---------------------------------------------------------------------------

def _firth_penalised_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    loglik = float(np.sum(y * (-np.logaddexp(0.0, -eta)) + (1.0 - y) * (-np.logaddexp(0.0, eta))))
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
    w = np.clip(p * (1.0 - p), 1e-10, None)
    info = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return loglik + 0.5 * float(logdet)


def firth_logistic(
    df: pd.DataFrame,
    *,
    y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    max_iter: int = 250,
    tol: float = 1e-8,
) -> AnalysisResult:
    """Firth bias-reduced logistic regression.

    Uses the Jeffreys-prior penalised score, which remains finite under many
    complete/quasi-separation configurations where ordinary maximum-likelihood
    logit diverges. Outcome must be binary after complete-case filtering.
    """
    if y not in df:
        raise ValueError(f"Outcome {y!r} is not present.")
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    yy = pd.to_numeric(df[y], errors="coerce")
    joined = pd.concat([yy.rename("__y__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < max(10, X.shape[1] + 3):
        raise ValueError("Too few complete observations for Firth logistic regression.")
    yv = joined.pop("__y__").to_numpy(float)
    unique = np.unique(yv)
    if len(unique) != 2:
        raise ValueError("Firth logistic regression requires a binary outcome with exactly two observed levels.")
    if not np.array_equal(unique, np.array([0.0, 1.0])):
        mapping = {unique[0]: 0.0, unique[1]: 1.0}
        yv = np.vectorize(mapping.get)(yv).astype(float)
    Xv = joined.to_numpy(float)
    beta = np.zeros(Xv.shape[1], dtype=float)
    converged = False
    iterations = 0
    current = _firth_penalised_loglik(beta, Xv, yv)
    for iteration in range(1, int(max_iter) + 1):
        eta = Xv @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        w = np.clip(p * (1.0 - p), 1e-10, None)
        info = Xv.T @ (w[:, None] * Xv)
        inv_info = np.linalg.pinv(info)
        # Diagonal of W^(1/2) X I^-1 X' W^(1/2)
        X_inv = Xv @ inv_info
        h = np.clip(w * np.einsum("ij,ij->i", X_inv, Xv), 0.0, 1.0)
        adjusted_score = Xv.T @ (yv - p + h * (0.5 - p))
        step = inv_info @ adjusted_score
        if not np.all(np.isfinite(step)):
            break
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + scale * step
            value = _firth_penalised_loglik(candidate, Xv, yv)
            if np.isfinite(value) and value >= current - 1e-12:
                beta = candidate
                current = value
                accepted = True
                break
            scale *= 0.5
        iterations = iteration
        if not accepted:
            break
        if np.max(np.abs(scale * step)) < tol:
            converged = True
            break
    eta = Xv @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
    w = np.clip(p * (1.0 - p), 1e-10, None)
    info = Xv.T @ (w[:, None] * Xv)
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2.0 * stats.norm.sf(np.abs(z))
    lo = beta - 1.959963984540054 * se
    hi = beta + 1.959963984540054 * se
    # Penalised-likelihood-ratio tests are slower than the Wald approximation but
    # materially useful in rare-event/separation settings.  For each term, fit
    # the nested Firth model with that column removed.
    plr_p = []
    plr_stat = []
    for j in range(Xv.shape[1]):
        Xr = np.delete(Xv, j, axis=1)
        if Xr.shape[1] == 0:
            plr_stat.append(np.nan); plr_p.append(np.nan); continue
        red = optimize.minimize(
            lambda b: -_firth_penalised_loglik(np.asarray(b), Xr, yv),
            np.zeros(Xr.shape[1]), method="BFGS",
            options={"maxiter": 1500, "gtol": 1e-8},
        )
        if np.isfinite(red.fun):
            lr = max(0.0, 2.0 * (current - (-float(red.fun))))
            plr_stat.append(lr); plr_p.append(float(stats.chi2.sf(lr, 1)))
        else:
            plr_stat.append(np.nan); plr_p.append(np.nan)
    coef = pd.DataFrame({
        "term": joined.columns.astype(str),
        "coefficient": beta,
        "std_error": se,
        "z": z,
        "p_value_wald": pvals,
        "penalised_LR_chi_square": plr_stat,
        "p_value_penalised_LR": plr_p,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "odds_ratio": np.exp(beta),
        "or_ci_95_low": np.exp(lo),
        "or_ci_95_high": np.exp(hi),
    })
    fitted = pd.DataFrame({
        "row_index": joined.index,
        "observed": yv,
        "fitted_probability": p,
        "pearson_residual": (yv - p) / np.sqrt(np.clip(p * (1 - p), 1e-10, None)),
    })
    fit = pd.DataFrame([{
        "n": len(yv),
        "parameters": len(beta),
        "penalised_log_likelihood": current,
        "converged": converged,
        "iterations": iterations,
        "events": int(yv.sum()),
        "non_events": int(len(yv) - yv.sum()),
    }])
    diagnostics = pd.DataFrame([
        {"diagnostic": "Convergence", "value": bool(converged), "detail": f"{iterations} iterations"},
        {"diagnostic": "Condition number", "value": float(np.linalg.cond(info)), "detail": "Large values indicate collinearity/scaling concerns."},
        {"diagnostic": "Probability range", "value": f"{p.min():.6g}–{p.max():.6g}", "detail": "Firth estimates can remain finite under separation."},
    ])
    warns: list[str] = []
    if not converged:
        warns.append("Firth iteration did not meet the requested tolerance; inspect scaling/collinearity and consider fewer predictors.")
    return AnalysisResult(
        {"Coefficients": coef, "Fit": fit, "Predictions": fitted}, diagnostics,
        {"method": "Firth logistic", "outcome": y, "predictors": list(x_vars), "categorical": list(categorical), "tol": tol, "max_iter": max_iter},
        warns,
    )


# ---------------------------------------------------------------------------
# 2) Ordered / multinomial / bounded / censored / zero-inflated models
# ---------------------------------------------------------------------------

def ordered_regression(
    df: pd.DataFrame,
    *,
    y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    distribution: str = "logit",
    category_order: Sequence[Any] | None = None,
) -> AnalysisResult:
    if distribution not in {"logit", "probit"}:
        raise ValueError("Ordered regression distribution must be 'logit' or 'probit'.")
    if y not in df:
        raise ValueError(f"Outcome {y!r} is not present.")
    X = _design_matrix(df, x_vars, categorical, add_constant=False)
    raw_y = df[y]
    if category_order is not None:
        cat = pd.Categorical(raw_y, categories=list(category_order), ordered=True)
        yy = pd.Series(cat.codes, index=df.index).replace(-1, np.nan)
        labels = list(category_order)
    else:
        numeric_y = pd.to_numeric(raw_y, errors="coerce")
        if numeric_y.notna().sum() >= raw_y.notna().sum() * 0.95:
            observed = sorted(numeric_y.dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(observed)}
            yy = numeric_y.map(mapping)
            labels = observed
        else:
            observed = list(pd.Series(raw_y.dropna().astype(str)).drop_duplicates())
            cat = pd.Categorical(raw_y.astype("string"), categories=observed, ordered=True)
            yy = pd.Series(cat.codes, index=df.index).replace(-1, np.nan)
            labels = observed
    joined = pd.concat([yy.rename("__y__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if joined["__y__"].nunique() < 3:
        raise ValueError("Ordered regression requires at least three observed outcome categories.")
    model = OrderedModel(joined["__y__"].astype(int), joined.drop(columns="__y__"), distr=distribution)
    result = model.fit(method="bfgs", disp=False, maxiter=2000)
    names = list(result.params.index)
    k_x = joined.shape[1] - 1
    coef = pd.DataFrame({
        "term": names,
        "coefficient": np.asarray(result.params),
        "std_error": np.asarray(result.bse),
        "p_value": np.asarray(result.pvalues),
    })
    ci = np.asarray(result.conf_int())
    coef["ci_95_low"] = ci[:, 0]
    coef["ci_95_high"] = ci[:, 1]
    coef["parameter_type"] = ["slope" if i < k_x else "threshold" for i in range(len(coef))]
    slope = coef.parameter_type.eq("slope")
    if distribution == "logit":
        coef.loc[slope, "odds_ratio"] = np.exp(coef.loc[slope, "coefficient"])
        coef.loc[slope, "or_ci_95_low"] = np.exp(coef.loc[slope, "ci_95_low"])
        coef.loc[slope, "or_ci_95_high"] = np.exp(coef.loc[slope, "ci_95_high"])
    pred = np.asarray(model.predict(result.params, exog=joined.drop(columns="__y__")))
    pred_cols = [f"P({label})" for label in labels[:pred.shape[1]]]
    predictions = pd.DataFrame(pred, index=joined.index, columns=pred_cols)
    predictions.insert(0, "observed_code", joined["__y__"].astype(int).to_numpy())
    predictions.insert(0, "row_index", joined.index)
    fit = pd.DataFrame([{
        "n": int(result.nobs), "categories": int(joined["__y__"].nunique()),
        "distribution": distribution, "log_likelihood": float(result.llf),
        "aic": float(result.aic), "bic": float(getattr(result, "bic", np.nan)),
        "converged": bool(result.mle_retvals.get("converged", True)),
    }])
    diagnostics = pd.DataFrame([
        {"diagnostic": "Observed categories", "value": len(labels), "detail": "; ".join(map(str, labels))},
        {"diagnostic": "Convergence", "value": bool(result.mle_retvals.get("converged", True)), "detail": str(result.mle_retvals)},
    ])
    return AnalysisResult(
        {"Coefficients": coef, "Fit": fit, "Predicted probabilities": predictions}, diagnostics,
        {"method": f"Ordered {distribution}", "outcome": y, "predictors": list(x_vars), "categorical": list(categorical), "category_order": labels},
        [], result,
    )


def brant_type_wald(
    df: pd.DataFrame,
    *,
    y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    category_order: Sequence[Any] | None = None,
) -> AnalysisResult:
    """Approximate Brant-style proportional-odds Wald diagnostic.

    Fits cumulative binary logits at every threshold and tests coefficient
    equality across thresholds. The implementation uses a block-diagonal
    covariance approximation, so it is deliberately labelled *Brant-type*
    rather than claiming exact equivalence to R/Stata Brant implementations.
    """
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    raw_y = df[y]
    if category_order is not None:
        cat = pd.Categorical(raw_y, categories=list(category_order), ordered=True)
        yy = pd.Series(cat.codes, index=df.index).replace(-1, np.nan)
        labels = list(category_order)
    else:
        numeric_y = pd.to_numeric(raw_y, errors="coerce")
        if numeric_y.notna().sum() >= raw_y.notna().sum() * .95:
            labels = sorted(numeric_y.dropna().unique().tolist())
            yy = numeric_y.map({v: i for i, v in enumerate(labels)})
        else:
            labels = list(pd.Series(raw_y.dropna().astype(str)).drop_duplicates())
            cat = pd.Categorical(raw_y.astype("string"), categories=labels, ordered=True)
            yy = pd.Series(cat.codes, index=df.index).replace(-1, np.nan)
    joined = pd.concat([yy.rename("__y__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    levels = sorted(joined["__y__"].astype(int).unique())
    if len(levels) < 3:
        raise ValueError("Brant-type diagnostic requires an ordinal outcome with at least three categories.")
    fits = []
    threshold_rows = []
    for threshold in levels[:-1]:
        binary = (joined["__y__"].astype(int) > threshold).astype(int)
        try:
            fit = sm.GLM(binary, joined.drop(columns="__y__"), family=sm.families.Binomial()).fit()
        except Exception as exc:
            raise ValueError(f"Threshold logit failed at threshold {threshold}: {exc}") from exc
        fits.append(fit)
        for term, b, se, p in zip(fit.params.index, fit.params, fit.bse, fit.pvalues):
            threshold_rows.append({"threshold_code": threshold, "threshold_label": str(labels[threshold]) if threshold < len(labels) else str(threshold), "term": term, "coefficient": b, "std_error": se, "p_value": p})
    rows = []
    for term in joined.drop(columns="__y__").columns:
        b = np.array([float(f.params[term]) for f in fits])
        v = np.array([float(f.cov_params().loc[term, term]) for f in fits])
        if len(b) < 2 or np.any(v <= 0):
            continue
        # Weighted equality-to-common-coefficient test.
        w = 1.0 / v
        mean_b = float(np.sum(w * b) / np.sum(w))
        q = float(np.sum(w * (b - mean_b) ** 2))
        df_q = len(b) - 1
        pval = float(stats.chi2.sf(q, df_q))
        rows.append({"term": term, "wald_chi_square": q, "df": df_q, "p_value": pval, "common_coefficient": mean_b, "thresholds": len(b)})
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["p_adjusted_bh"] = multipletests(tests.p_value, method="fdr_bh")[1]
    diagnostics = pd.DataFrame([
        {"diagnostic": "Procedure", "value": "Brant-type approximate Wald", "detail": "Cumulative logits + block-diagonal threshold covariance approximation."},
        {"diagnostic": "Threshold models", "value": len(fits), "detail": "One binary logit per cumulative split."},
    ])
    return AnalysisResult(
        {"Threshold coefficients": pd.DataFrame(threshold_rows), "Proportional-odds tests": tests}, diagnostics,
        {"method": "Brant-type Wald", "outcome": y, "predictors": list(x_vars), "category_order": labels},
        ["This is a transparent Brant-type approximation, not a claim of numerical identity with every Brant implementation."],
    )


def multinomial_logit(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    raw = df[y]
    cat = pd.Categorical(raw)
    yy = pd.Series(cat.codes, index=df.index).replace(-1, np.nan)
    joined = pd.concat([yy.rename("__y__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if joined["__y__"].nunique() < 3:
        raise ValueError("Multinomial logit requires at least three outcome categories.")
    result = sm.MNLogit(joined["__y__"].astype(int), joined.drop(columns="__y__")).fit(method="newton", maxiter=200, disp=False)
    params = result.params
    bse = result.bse
    pvals = result.pvalues
    rows = []
    for outcome_col in params.columns:
        label_idx = int(outcome_col) + 1 if str(outcome_col).isdigit() else outcome_col
        label = cat.categories[label_idx] if isinstance(label_idx, int) and label_idx < len(cat.categories) else label_idx
        for term in params.index:
            b = float(params.loc[term, outcome_col])
            se = float(bse.loc[term, outcome_col])
            rows.append({"outcome_vs_base": str(label), "term": term, "coefficient": b, "std_error": se, "p_value": float(pvals.loc[term, outcome_col]), "relative_risk_ratio": float(np.exp(b)), "rrr_ci_95_low": float(np.exp(b - 1.95996398454 * se)), "rrr_ci_95_high": float(np.exp(b + 1.95996398454 * se))})
    fit = pd.DataFrame([{"n": int(result.nobs), "categories": int(joined["__y__"].nunique()), "log_likelihood": float(result.llf), "aic": float(result.aic), "bic": float(result.bic), "pseudo_r_squared": float(result.prsquared)}])
    return AnalysisResult({"Coefficients": pd.DataFrame(rows), "Fit": fit}, pd.DataFrame(), {"method": "Multinomial logit", "outcome": y, "predictors": list(x_vars)}, [], result)


def beta_regression(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    precision_vars: Sequence[str] = (),
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    Z = _design_matrix(df, precision_vars, (), add_constant=True) if precision_vars else pd.DataFrame({"precision_const": 1.0}, index=df.index)
    yy = pd.to_numeric(df[y], errors="coerce")
    joined = pd.concat([yy.rename("__y__"), X.add_prefix("mean::"), Z.add_prefix("precision::")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if (joined["__y__"] <= 0).any() or (joined["__y__"] >= 1).any():
        raise ValueError("Beta regression requires all analysed outcomes strictly inside (0, 1). Use an explicit boundary transformation only if substantively justified.")
    mean_cols = [c for c in joined if c.startswith("mean::")]
    prec_cols = [c for c in joined if c.startswith("precision::")]
    model = BetaModel(joined["__y__"], joined[mean_cols].rename(columns=lambda c: c.replace("mean::", "")), exog_precision=joined[prec_cols].rename(columns=lambda c: c.replace("precision::", "")))
    result = model.fit(disp=False, maxiter=1000)
    coef = _coef_table(result)
    fit = pd.DataFrame([{"n": int(result.nobs), "log_likelihood": float(result.llf), "aic": float(result.aic), "bic": float(result.bic), "converged": bool(result.mle_retvals.get("converged", True))}])
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, pd.DataFrame(), {"method": "Beta regression", "outcome": y, "predictors": list(x_vars), "precision_predictors": list(precision_vars)}, [], result)


def tobit_regression(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    lower: float | None = None,
    upper: float | None = None,
) -> AnalysisResult:
    """Two-sided censored normal (Tobit) maximum-likelihood regression."""
    if lower is None and upper is None:
        raise ValueError("Specify at least one censoring bound for Tobit regression.")
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    yy = pd.to_numeric(df[y], errors="coerce")
    joined = pd.concat([yy.rename("__y__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    yv = joined.pop("__y__").to_numpy(float)
    Xv = joined.to_numpy(float)
    ols = sm.OLS(yv, Xv).fit()
    start = np.r_[np.asarray(ols.params), np.log(max(np.std(ols.resid, ddof=max(1, Xv.shape[1])), 1e-6))]

    def nll(par: np.ndarray) -> float:
        beta = par[:-1]
        sigma = np.exp(par[-1])
        mu = Xv @ beta
        ll = np.zeros(len(yv))
        unc = np.ones(len(yv), dtype=bool)
        if lower is not None:
            lc = yv <= float(lower) + 1e-12
            ll[lc] = stats.norm.logcdf((float(lower) - mu[lc]) / sigma)
            unc &= ~lc
        if upper is not None:
            uc = yv >= float(upper) - 1e-12
            ll[uc] = stats.norm.logsf((float(upper) - mu[uc]) / sigma)
            unc &= ~uc
        ll[unc] = stats.norm.logpdf((yv[unc] - mu[unc]) / sigma) - np.log(sigma)
        if np.any(~np.isfinite(ll)):
            return 1e100
        return -float(ll.sum())

    res = optimize.minimize(nll, start, method="BFGS", options={"maxiter": 2000, "gtol": 1e-7})
    par = np.asarray(res.x)
    hess = approx_hess(par, nll)
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    names = list(joined.columns) + ["log_sigma"]
    coef = pd.DataFrame({"term": names, "coefficient": par, "std_error": se})
    coef["z"] = coef.coefficient / coef.std_error
    coef["p_value"] = 2 * stats.norm.sf(np.abs(coef.z))
    coef["ci_95_low"] = coef.coefficient - 1.95996398454 * coef.std_error
    coef["ci_95_high"] = coef.coefficient + 1.95996398454 * coef.std_error
    fit = pd.DataFrame([{"n": len(yv), "log_likelihood": -float(res.fun), "aic": 2 * len(par) + 2 * float(res.fun), "bic": len(par) * np.log(len(yv)) + 2 * float(res.fun), "converged": bool(res.success), "lower_bound": lower, "upper_bound": upper, "sigma": float(np.exp(par[-1]))}])
    diagnostics = pd.DataFrame([{"diagnostic": "Optimizer", "value": str(res.message), "detail": f"iterations={getattr(res, 'nit', np.nan)}"}])
    warns = [] if res.success else ["Tobit optimiser did not report successful convergence."]
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, diagnostics, {"method": "Tobit censored normal", "outcome": y, "lower": lower, "upper": upper}, warns)


def zero_inflated_count(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    inflation_vars: Sequence[str] = (),
    categorical: Sequence[str] = (),
    model: str = "ZIP",
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    Zi = _design_matrix(df, inflation_vars or x_vars, categorical, add_constant=True)
    yy = pd.to_numeric(df[y], errors="coerce")
    joined = pd.concat([yy.rename("__y__"), X.add_prefix("count::"), Zi.add_prefix("infl::")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if (joined["__y__"] < 0).any() or not np.allclose(joined["__y__"], np.round(joined["__y__"])):
        raise ValueError("Zero-inflated count models require non-negative integer outcomes.")
    count_cols = [c for c in joined if c.startswith("count::")]
    infl_cols = [c for c in joined if c.startswith("infl::")]
    klass = ZeroInflatedNegativeBinomialP if model.upper() in {"ZINB", "ZINB-P"} else ZeroInflatedPoisson
    mod = klass(joined["__y__"].astype(int), joined[count_cols].rename(columns=lambda c: c.replace("count::", "")), exog_infl=joined[infl_cols].rename(columns=lambda c: c.replace("infl::", "")), inflation="logit")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mod.fit(method="bfgs", maxiter=1000, disp=0)
    coef = _coef_table(result, exponentiate=True, effect_name="exp_coefficient")
    fit = pd.DataFrame([{"n": int(result.nobs), "model": model.upper(), "log_likelihood": float(result.llf), "aic": float(result.aic), "bic": float(getattr(result, "bic", np.nan)), "zero_fraction": float((joined["__y__"] == 0).mean())}])
    warns = [str(w.message) for w in caught]
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, pd.DataFrame(), {"method": model.upper(), "outcome": y, "count_predictors": list(x_vars), "inflation_predictors": list(inflation_vars or x_vars)}, warns, result)


# ---------------------------------------------------------------------------
# 3) Clustered/repeated/survival models
# ---------------------------------------------------------------------------

def linear_mixed_effects(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    group: str,
    categorical: Sequence[str] = (),
    random_slope: str | None = None,
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    yy = pd.to_numeric(df[y], errors="coerce")
    groups = df[group]
    pieces = [yy.rename("__y__"), groups.rename("__group__"), X]
    if random_slope:
        rs = pd.to_numeric(df[random_slope], errors="coerce").rename("__rs__")
        pieces.append(rs)
    joined = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    exog_re = None
    if random_slope:
        exog_re = sm.add_constant(joined[["__rs__"]], has_constant="add")
    model = sm.MixedLM(joined["__y__"], joined[X.columns], groups=joined["__group__"], exog_re=exog_re)
    result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    coef = _coef_table(result)
    fit = pd.DataFrame([{"n": int(result.nobs), "groups": int(joined["__group__"].nunique()), "log_likelihood": float(result.llf), "aic": float(result.aic), "bic": float(result.bic), "converged": bool(result.converged)}])
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, pd.DataFrame(), {"method": "Linear mixed effects", "outcome": y, "group": group, "random_slope": random_slope}, [], result)


def gee_regression(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    group: str,
    categorical: Sequence[str] = (),
    family: str = "Gaussian",
    correlation: str = "Exchangeable",
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=True)
    yy = pd.to_numeric(df[y], errors="coerce")
    joined = pd.concat([yy.rename("__y__"), df[group].rename("__group__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    fam = {"Gaussian": Gaussian(), "Binomial": Binomial(), "Poisson": Poisson()}.get(family, Gaussian())
    cov_struct = Exchangeable() if correlation == "Exchangeable" else Independence()
    model = GEE(joined["__y__"], joined[X.columns], groups=joined["__group__"], family=fam, cov_struct=cov_struct)
    result = model.fit(maxiter=200)
    exponentiate = family in {"Binomial", "Poisson"}
    coef = _coef_table(result, exponentiate=exponentiate, effect_name="odds_or_rate_ratio")
    fit = pd.DataFrame([{"n": int(result.nobs), "clusters": int(joined["__group__"].nunique()), "family": family, "working_correlation": correlation, "scale": float(result.scale)}])
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, pd.DataFrame(), {"method": "GEE", "outcome": y, "group": group, "family": family, "correlation": correlation}, [], result)


def cox_proportional_hazards(
    df: pd.DataFrame,
    *, time: str,
    event: str,
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    strata: str | None = None,
) -> AnalysisResult:
    X = _design_matrix(df, x_vars, categorical, add_constant=False)
    tt = pd.to_numeric(df[time], errors="coerce")
    ee = pd.to_numeric(df[event], errors="coerce")
    pieces = [tt.rename("__time__"), ee.rename("__event__"), X]
    if strata:
        pieces.append(df[strata].rename("__strata__"))
    joined = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if (joined["__time__"] <= 0).any():
        raise ValueError("Cox PH requires strictly positive follow-up times.")
    stratum = joined["__strata__"] if strata else None
    model = PHReg(joined["__time__"], joined[X.columns], status=joined["__event__"], strata=stratum, ties="efron")
    result = model.fit()
    names = list(X.columns)
    b = np.asarray(result.params)
    se = np.asarray(result.bse)
    p = np.asarray(result.pvalues)
    coef = pd.DataFrame({"term": names, "coefficient": b, "std_error": se, "p_value": p, "hazard_ratio": np.exp(b), "hr_ci_95_low": np.exp(b - 1.95996398454 * se), "hr_ci_95_high": np.exp(b + 1.95996398454 * se)})
    fit = pd.DataFrame([{"n": len(joined), "events": int(joined["__event__"].sum()), "log_likelihood": float(result.llf), "stratified": bool(strata)}])
    diagnostics = pd.DataFrame([{"diagnostic": "PH assumption", "value": "Not automatically proven", "detail": "Inspect time-varying effects/Schoenfeld-style diagnostics when publication requires a formal proportional-hazards assessment."}])
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, diagnostics, {"method": "Cox proportional hazards", "time": time, "event": event, "strata": strata}, [], result)


# ---------------------------------------------------------------------------
# 4) Compositional data: zero replacement, CLR/ILR, PERMANOVA, Dirichlet
# ---------------------------------------------------------------------------

def _prepare_composition(values: np.ndarray, zero_replacement: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("A composition requires at least two component columns.")
    if np.any(arr < 0):
        raise ValueError("Compositional values cannot be negative.")
    if zero_replacement <= 0:
        if np.any(arr <= 0):
            raise ValueError("Log-ratio transforms require strictly positive components; specify a positive zero replacement.")
        work = arr.copy()
    else:
        work = np.where(arr <= 0, float(zero_replacement), arr)
    sums = work.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError("Every composition row must have a positive total.")
    return work / sums[:, None]


def clr_transform(values: np.ndarray, zero_replacement: float = 1e-6) -> np.ndarray:
    comp = _prepare_composition(values, zero_replacement)
    logs = np.log(comp)
    return logs - logs.mean(axis=1, keepdims=True)


def ilr_transform(values: np.ndarray, zero_replacement: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    comp = _prepare_composition(values, zero_replacement)
    basis = linalg.helmert(comp.shape[1], full=False)  # (D-1) x D, orthonormal rows
    coords = np.log(comp) @ basis.T
    return coords, basis


def compositional_transforms(
    df: pd.DataFrame,
    *, columns: Sequence[str],
    zero_replacement: float = 1e-6,
) -> AnalysisResult:
    data = _as_numeric_frame(df, columns).dropna()
    comp = _prepare_composition(data.to_numpy(float), zero_replacement)
    clr = clr_transform(comp, 0)
    ilr, basis = ilr_transform(comp, 0)
    normalized = pd.DataFrame(comp, index=data.index, columns=columns)
    clr_df = pd.DataFrame(clr, index=data.index, columns=[f"CLR::{c}" for c in columns])
    ilr_df = pd.DataFrame(ilr, index=data.index, columns=[f"ILR{i+1}" for i in range(ilr.shape[1])])
    basis_df = pd.DataFrame(basis, columns=columns, index=[f"ILR{i+1}" for i in range(basis.shape[0])]).reset_index(names="coordinate")
    diagnostics = pd.DataFrame([{"diagnostic": "Rows analysed", "value": len(data), "detail": f"zero replacement={zero_replacement:g}"}])
    return AnalysisResult({"Closed composition": normalized.reset_index(names="row_index"), "CLR coordinates": clr_df.reset_index(names="row_index"), "ILR coordinates": ilr_df.reset_index(names="row_index"), "ILR basis": basis_df}, diagnostics, {"method": "CLR/ILR transforms", "columns": list(columns), "zero_replacement": zero_replacement}, [])


def permanova(
    df: pd.DataFrame,
    *, columns: Sequence[str],
    group: str,
    transform: str = "ILR (Aitchison)",
    permutations: int = 999,
    seed: int = 42,
    zero_replacement: float = 1e-6,
) -> AnalysisResult:
    work = pd.concat([_as_numeric_frame(df, columns), df[group].rename("__group__")], axis=1).dropna()
    if work["__group__"].nunique() < 2:
        raise ValueError("PERMANOVA requires at least two groups.")
    arr = work[list(columns)].to_numpy(float)
    if transform.startswith("ILR"):
        X, _ = ilr_transform(arr, zero_replacement)
    elif transform.startswith("CLR"):
        X = clr_transform(arr, zero_replacement)
    else:
        X = arr
    labels = work["__group__"].astype(str).to_numpy()
    unique = np.unique(labels)
    n, p = X.shape
    grand = X.mean(axis=0)
    ss_total = float(np.square(X - grand).sum())

    def statistic(lbl: np.ndarray) -> tuple[float, float, float]:
        ss_between = 0.0
        for g in np.unique(lbl):
            Xi = X[lbl == g]
            ss_between += len(Xi) * float(np.square(Xi.mean(axis=0) - grand).sum())
        ss_within = max(ss_total - ss_between, 0.0)
        df_b = len(np.unique(lbl)) - 1
        df_w = n - len(np.unique(lbl))
        f = (ss_between / df_b) / (ss_within / df_w) if df_b > 0 and df_w > 0 and ss_within > 0 else np.nan
        r2 = ss_between / ss_total if ss_total > 0 else np.nan
        return float(f), float(ss_between), float(r2)

    observed, ss_b, r2 = statistic(labels)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(int(permutations), dtype=float)
    for i in range(int(permutations)):
        perm_stats[i] = statistic(rng.permutation(labels))[0]
    p_value = float((1 + np.sum(perm_stats >= observed - 1e-15)) / (len(perm_stats) + 1))
    table = pd.DataFrame([{"pseudo_F": observed, "p_value_permutation": p_value, "R_squared": r2, "SS_between": ss_b, "SS_total": ss_total, "groups": len(unique), "n": n, "dimensions": p, "permutations": int(permutations)}])
    perm = pd.DataFrame({"permutation": np.arange(1, len(perm_stats)+1), "pseudo_F": perm_stats})
    diagnostics = pd.DataFrame([{"diagnostic": "Geometry", "value": transform, "detail": "ILR makes Euclidean distance equivalent to Aitchison geometry for closed positive compositions."}])
    return AnalysisResult({"PERMANOVA": table, "Permutation distribution": perm}, diagnostics, {"method": "PERMANOVA", "columns": list(columns), "group": group, "transform": transform, "permutations": permutations, "seed": seed}, [])


def dirichlet_regression(
    df: pd.DataFrame,
    *, components: Sequence[str],
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    reference_levels: Mapping[str, Any] | None = None,
    zero_replacement: float = 1e-6,
    max_iter: int = 2000,
) -> AnalysisResult:
    """Dirichlet regression with multinomial-logit mean and common precision."""
    Yraw = _as_numeric_frame(df, components)
    X = _design_matrix(df, x_vars, categorical, add_constant=True, reference_levels=reference_levels)
    joined = pd.concat([Yraw.add_prefix("y::"), X.add_prefix("x::")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_cols = [c for c in joined if c.startswith("y::")]
    x_cols = [c for c in joined if c.startswith("x::")]
    Y = _prepare_composition(joined[y_cols].to_numpy(float), zero_replacement)
    Xv = joined[x_cols].to_numpy(float)
    n, k = Y.shape
    p = Xv.shape[1]
    if n < max(20, p + k + 5):
        raise ValueError("Too few complete rows for the selected Dirichlet specification.")

    def unpack(par: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        B = par[: p * (k - 1)].reshape(p, k - 1)
        phi = float(np.exp(np.clip(par[-1], -10, 15)))
        eta = Xv @ B
        eta_full = np.column_stack([eta, np.zeros(n)])
        eta_full -= eta_full.max(axis=1, keepdims=True)
        exp_eta = np.exp(eta_full)
        mu = exp_eta / exp_eta.sum(axis=1, keepdims=True)
        return B, phi, mu

    def nll(par: np.ndarray) -> float:
        _, phi, mu = unpack(par)
        alpha = np.clip(mu * phi, 1e-10, None)
        ll = gammaln(phi) - gammaln(alpha).sum(axis=1) + ((alpha - 1.0) * np.log(np.clip(Y, 1e-300, None))).sum(axis=1)
        if not np.all(np.isfinite(ll)):
            return 1e100
        return -float(ll.sum())

    start = np.zeros(p * (k - 1) + 1)
    start[-1] = np.log(20.0)
    res = optimize.minimize(nll, start, method="BFGS", options={"maxiter": int(max_iter), "gtol": 1e-6})
    par = np.asarray(res.x)
    H = approx_hess(par, nll)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    rows = []
    idx = 0
    predictor_names = [c.replace("x::", "") for c in x_cols]
    for j, component in enumerate(components[:-1]):
        for predictor in predictor_names:
            b = par[idx]; s = se[idx]
            rows.append({"component_vs_baseline": component, "baseline_component": components[-1], "term": predictor, "coefficient": b, "std_error": s, "z": b/s if s > 0 else np.nan, "p_value": 2*stats.norm.sf(abs(b/s)) if s > 0 else np.nan, "ci_95_low": b-1.95996398454*s, "ci_95_high": b+1.95996398454*s})
            idx += 1
    rows.append({"component_vs_baseline": "PRECISION", "baseline_component": "", "term": "log_phi", "coefficient": par[-1], "std_error": se[-1], "z": par[-1]/se[-1] if se[-1]>0 else np.nan, "p_value": 2*stats.norm.sf(abs(par[-1]/se[-1])) if se[-1]>0 else np.nan, "ci_95_low": par[-1]-1.95996398454*se[-1], "ci_95_high": par[-1]+1.95996398454*se[-1]})
    B, phi, mu = unpack(par)
    mean_pred = pd.DataFrame(mu, index=joined.index, columns=[f"predicted_mean::{c}" for c in components]).reset_index(names="row_index")
    fit = pd.DataFrame([{"n": n, "components": k, "predictors_including_intercept": p, "precision_phi": phi, "log_likelihood": -float(res.fun), "aic": 2*len(par)+2*float(res.fun), "bic": len(par)*np.log(n)+2*float(res.fun), "converged": bool(res.success), "optimizer_message": str(res.message)}])
    warns = [] if res.success else ["Dirichlet optimiser did not report successful convergence; inspect zeros, scaling and model complexity."]
    return AnalysisResult({"Coefficients": pd.DataFrame(rows), "Fit": fit, "Predicted compositions": mean_pred}, pd.DataFrame(), {"method": "Dirichlet regression (mean/precision)", "components": list(components), "predictors": list(x_vars), "categorical": list(categorical), "reference_levels": dict(reference_levels or {}), "zero_replacement": zero_replacement}, warns)


def dirichlet_component_alpha_regression(
    df: pd.DataFrame,
    *,
    components: Sequence[str],
    x_vars: Sequence[str],
    categorical: Sequence[str] = (),
    reference_levels: Mapping[str, Any] | None = None,
    standardize_numeric: Sequence[str] = (),
    zero_replacement: float = 1e-6,
    max_iter: int = 3000,
    likelihood_ratio_blocks: bool = True,
) -> AnalysisResult:
    """Component-wise Dirichlet regression using log(alpha_j)=X beta_j.

    Unlike the alternative mean/precision parameterisation, every predictor is
    allowed to change every Dirichlet concentration parameter.  A categorical
    predictor with ``g-1`` dummy columns therefore contributes ``K*(g-1)``
    degrees of freedom to a likelihood-ratio block test for a K-part
    composition.  This is the parameterisation used by several established
    Dirichlet-regression workflows and is retained alongside, not instead of,
    the mean/precision model.
    """
    components = list(dict.fromkeys(components))
    x_vars = list(dict.fromkeys(x_vars))
    categorical = [c for c in categorical if c in x_vars]
    if len(components) < 2:
        raise ValueError("Dirichlet regression requires at least two composition components.")
    model_df = df.copy()
    standardize_numeric = [c for c in standardize_numeric if c in x_vars and c not in categorical]
    standardization_rows: list[dict[str, Any]] = []
    for col in standardize_numeric:
        values = pd.to_numeric(model_df[col], errors="coerce")
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot z-standardise {col!r}: population SD is zero or non-finite.")
        model_df[col] = (values - mean) / sd
        standardization_rows.append({"predictor": col, "mean": mean, "population_sd_ddof0": sd})
    Yraw = _as_numeric_frame(model_df, components)
    X = _design_matrix(
        model_df, x_vars, categorical, add_constant=True,
        reference_levels=reference_levels,
    )
    joined = pd.concat([Yraw.add_prefix("y::"), X.add_prefix("x::")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_cols = [c for c in joined if c.startswith("y::")]
    x_cols = [c for c in joined if c.startswith("x::")]
    Y = _prepare_composition(joined[y_cols].to_numpy(float), zero_replacement)
    Xv = joined[x_cols].to_numpy(float)
    n, k = Y.shape
    p = Xv.shape[1]
    if n < max(20, p + k + 5):
        raise ValueError("Too few complete rows for the selected component-wise Dirichlet specification.")
    predictor_names = [c.replace("x::", "") for c in x_cols]
    logY = np.log(np.clip(Y, 1e-300, None))

    def unpack(par: np.ndarray, design: np.ndarray = Xv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = np.asarray(par, dtype=float).reshape(design.shape[1], k)
        eta = np.clip(design @ B, -20.0, 20.0)
        alpha = np.exp(eta)
        alpha0 = alpha.sum(axis=1)
        mu = alpha / alpha0[:, None]
        return B, alpha, mu

    def make_objective(design: np.ndarray):
        def _nll(par: np.ndarray) -> float:
            B = np.asarray(par, dtype=float).reshape(design.shape[1], k)
            eta = np.clip(design @ B, -20.0, 20.0)
            alpha = np.exp(eta)
            alpha0 = alpha.sum(axis=1)
            ll = gammaln(alpha0) - gammaln(alpha).sum(axis=1) + ((alpha - 1.0) * logY).sum(axis=1)
            if not np.all(np.isfinite(ll)):
                return 1e100
            return -float(ll.sum())
        def _grad(par: np.ndarray) -> np.ndarray:
            B = np.asarray(par, dtype=float).reshape(design.shape[1], k)
            eta = np.clip(design @ B, -20.0, 20.0)
            alpha = np.exp(eta)
            alpha0 = alpha.sum(axis=1)
            d_ll_d_alpha = digamma(alpha0)[:, None] - digamma(alpha) + logY
            d_ll_d_eta = alpha * d_ll_d_alpha
            return -(design.T @ d_ll_d_eta).ravel()
        return _nll, _grad

    # Stable starting values: observed mean composition with moderate common
    # concentration.  Non-intercept slopes begin at zero.
    mean_y = np.clip(Y.mean(axis=0), 1e-6, None)
    mean_y /= mean_y.sum()
    B0 = np.zeros((p, k), dtype=float)
    const_idx = predictor_names.index("const") if "const" in predictor_names else None
    if const_idx is not None:
        B0[const_idx, :] = np.log(mean_y * 20.0)
    nll, grad = make_objective(Xv)
    res = optimize.minimize(
        nll, B0.ravel(), jac=grad, method="BFGS",
        options={"maxiter": int(max_iter), "gtol": 1e-7},
    )
    par = np.asarray(res.x, dtype=float)
    B, alpha, mu = unpack(par)
    full_ll = -float(res.fun)

    H = approx_hess(par, nll)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf)).reshape(p, k)
    rows: list[dict[str, Any]] = []
    for i, term in enumerate(predictor_names):
        for j, component in enumerate(components):
            b = float(B[i, j]); s = float(se[i, j])
            z = b / s if s > 0 else np.nan
            rows.append({
                "component": component, "term": term, "coefficient": b,
                "std_error": s, "z": z,
                "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                "ci_95_low": b - 1.95996398454 * s,
                "ci_95_high": b + 1.95996398454 * s,
                "alpha_ratio": float(np.exp(np.clip(b, -700, 700))),
            })

    block_rows: list[dict[str, Any]] = []
    if likelihood_ratio_blocks and x_vars:
        for var in x_vars:
            if var in categorical:
                idxs = [i for i, name in enumerate(predictor_names) if name.startswith(f"{var}_")]
            else:
                idxs = [i for i, name in enumerate(predictor_names) if name == var]
            if not idxs:
                continue
            keep = [i for i in range(p) if i not in idxs]
            Xr = Xv[:, keep]
            Br0 = B[keep, :].ravel()
            reduced_nll, reduced_grad = make_objective(Xr)
            red = optimize.minimize(
                reduced_nll, Br0, jac=reduced_grad, method="BFGS",
                options={"maxiter": int(max_iter), "gtol": 1e-7},
            )
            red_ll = -float(red.fun)
            lr = max(0.0, 2.0 * (full_ll - red_ll))
            df_lr = int(len(idxs) * k)
            block_rows.append({
                "predictor_block": var, "dummy_columns_removed": len(idxs),
                "components": k, "df": df_lr, "LR_chi_square": lr,
                "p_value": float(stats.chi2.sf(lr, df_lr)),
                "full_log_likelihood": full_ll, "reduced_log_likelihood": red_ll,
                "reduced_converged": bool(red.success or np.max(np.abs(reduced_grad(np.asarray(red.x)))) < 1e-5),
            })

    mean_pred = pd.DataFrame(mu, index=joined.index, columns=[f"predicted_mean::{c}" for c in components])
    mean_pred.insert(0, "predicted_precision", alpha.sum(axis=1))
    mean_pred = mean_pred.reset_index(names="row_index")
    alpha_pred = pd.DataFrame(alpha, index=joined.index, columns=[f"alpha::{c}" for c in components]).reset_index(names="row_index")
    n_params = p * k
    fit = pd.DataFrame([{
        "n": n, "components": k, "design_columns_including_intercept": p,
        "parameters": n_params, "log_likelihood": full_ll,
        "aic": 2 * n_params - 2 * full_ll,
        "bic": n_params * np.log(n) - 2 * full_ll,
        "converged": bool(res.success), "optimizer_message": str(res.message),
    }])
    diagnostics = pd.DataFrame([
        {"diagnostic": "Parameterisation", "value": "log(alpha_j)=X beta_j", "detail": "Each predictor can affect every composition component."},
        {"diagnostic": "Reference levels", "value": str(dict(reference_levels or {})), "detail": "Explicit references are used before dummy encoding."},
        {"diagnostic": "Complete rows", "value": n, "detail": f"{len(df)-n} rows excluded by complete-case filtering."},
    ])
    tables = {
        "Coefficients": pd.DataFrame(rows),
        "Likelihood-ratio block tests": pd.DataFrame(block_rows),
        "Fit": fit,
        "Standardisation": pd.DataFrame(standardization_rows),
        "Predicted compositions": mean_pred,
        "Predicted alpha": alpha_pred,
    }
    warns = [] if res.success else ["Component-wise Dirichlet optimiser did not report successful convergence; inspect scaling, zeros and model complexity."]
    return AnalysisResult(
        tables, diagnostics,
        {
            "method": "Dirichlet regression (component-wise log-alpha)",
            "components": components, "predictors": x_vars,
            "categorical": categorical, "reference_levels": dict(reference_levels or {}),
            "standardize_numeric": list(standardize_numeric),
            "standardization_ddof": 0,
            "zero_replacement": zero_replacement,
            "likelihood_ratio_blocks": bool(likelihood_ratio_blocks),
        },
        warns, res,
    )


# ---------------------------------------------------------------------------
# 5) Repeated ranks / Plackett-Luce / mixtures
# ---------------------------------------------------------------------------

def repeated_rank_tests(
    df: pd.DataFrame,
    *, columns: Sequence[str],
    higher_is_better: bool = True,
    adjustment: str = "holm",
) -> AnalysisResult:
    data = _as_numeric_frame(df, columns).dropna()
    if len(data) < 3 or len(columns) < 2:
        raise ValueError("At least three complete respondents and two repeated measures are required.")
    samples = [data[c].to_numpy(float) for c in columns]
    friedman = stats.friedmanchisquare(*samples)
    n, k = data.shape
    W = float(friedman.statistic / (n * (k - 1))) if k > 1 else np.nan
    pairs = []
    for a, b in combinations(columns, 2):
        try:
            result = stats.wilcoxon(data[a], data[b], alternative="two-sided", zero_method="wilcox", method="auto")
            stat, p = float(result.statistic), float(result.pvalue)
        except ValueError:
            stat, p = 0.0, 1.0
        diff = data[a] - data[b]
        pairs.append({"variable_a": a, "variable_b": b, "wilcoxon_W": stat, "p_value": p, "median_difference": float(np.median(diff)), "mean_difference": float(np.mean(diff))})
    pair_df = pd.DataFrame(pairs)
    if not pair_df.empty:
        pair_df["p_adjusted"] = multipletests(pair_df.p_value, method=adjustment)[1]
        pair_df["adjustment"] = adjustment
    rank_values = data.rank(axis=1, ascending=not higher_is_better, method="average")
    rank_summary = pd.DataFrame({"variable": columns, "mean_rank": rank_values.mean(axis=0).to_numpy(), "median_rank": rank_values.median(axis=0).to_numpy()}).sort_values("mean_rank")
    omnibus = pd.DataFrame([{"n": n, "conditions": k, "friedman_chi_square": float(friedman.statistic), "df": k-1, "p_value": float(friedman.pvalue), "kendall_W": W}])
    return AnalysisResult({"Omnibus": omnibus, "Pairwise Wilcoxon": pair_df, "Rank summary": rank_summary}, pd.DataFrame(), {"method": "Friedman + Wilcoxon", "columns": list(columns), "adjustment": adjustment}, [])


def _rank_orders(rank_matrix: np.ndarray) -> list[np.ndarray]:
    orders: list[np.ndarray] = []
    for row in rank_matrix:
        if np.any(~np.isfinite(row)):
            continue
        # Stable sort provides deterministic handling of accidental ties; caller is warned.
        orders.append(np.argsort(row, kind="mergesort"))
    return orders


def _pl_logprob_order(order: np.ndarray, theta: np.ndarray) -> float:
    worth = np.exp(np.r_[theta, 0.0])
    ll = 0.0
    remaining = list(order.astype(int))
    while len(remaining) > 1:
        chosen = remaining[0]
        denom = worth[remaining].sum()
        ll += math.log(max(worth[chosen], 1e-300)) - math.log(max(denom, 1e-300))
        remaining.pop(0)
    return float(ll)


def _fit_pl_component(orders: list[np.ndarray], weights: np.ndarray | None = None, start: np.ndarray | None = None) -> optimize.OptimizeResult:
    if not orders:
        raise ValueError("No complete rankings are available.")
    k = len(orders[0])
    if weights is None:
        weights = np.ones(len(orders))
    weights = np.asarray(weights, dtype=float)
    if start is None:
        start = np.zeros(k - 1)

    def nll(theta: np.ndarray) -> float:
        vals = np.array([_pl_logprob_order(order, theta) for order in orders])
        return -float(np.sum(weights * vals))

    return optimize.minimize(nll, np.asarray(start), method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})


def plackett_luce(
    df: pd.DataFrame,
    *, rank_columns: Sequence[str],
) -> AnalysisResult:
    data = _as_numeric_frame(df, rank_columns).dropna()
    orders = _rank_orders(data.to_numpy(float))
    if len(orders) < 5:
        raise ValueError("At least five complete rankings are required.")
    ties = int((data.nunique(axis=1) < len(rank_columns)).sum())
    res = _fit_pl_component(orders)
    theta = np.asarray(res.x)
    worth = np.exp(np.r_[theta, 0.0]); worth /= worth.sum()
    hess = approx_hess(theta, lambda t: -sum(_pl_logprob_order(o, t) for o in orders))
    cov = np.linalg.pinv(hess)
    se_theta = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    worth_table = pd.DataFrame({"item": rank_columns, "worth": worth, "log_worth_vs_baseline": np.r_[theta, 0.0], "std_error_log_worth": np.r_[se_theta, np.nan]}).sort_values("worth", ascending=False)
    ll = -float(res.fun); params = len(theta)
    fit = pd.DataFrame([{"n_rankings": len(orders), "items": len(rank_columns), "log_likelihood": ll, "aic": 2*params-2*ll, "bic": params*np.log(len(orders))-2*ll, "converged": bool(res.success), "tied_rank_rows": ties}])
    warns = ["Tied ranks were deterministically ordered for the Plackett–Luce likelihood; use a dedicated tie model if ties are substantively meaningful."] if ties else []
    return AnalysisResult({"Worths": worth_table, "Fit": fit}, pd.DataFrame(), {"method": "Plackett-Luce", "rank_columns": list(rank_columns)}, warns)


def plackett_luce_mixture(
    df: pd.DataFrame,
    *, rank_columns: Sequence[str],
    components: int = 2,
    seed: int = 42,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> AnalysisResult:
    data = _as_numeric_frame(df, rank_columns).dropna()
    orders = _rank_orders(data.to_numpy(float))
    n = len(orders); k_items = len(rank_columns); K = int(components)
    if K < 2 or K > 5:
        raise ValueError("Plackett–Luce mixture supports 2–5 components in the interactive workbench.")
    if n < max(20, K * 8):
        raise ValueError("Too few complete rankings for the requested mixture size.")
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.15, size=(K, k_items - 1))
    pi = np.ones(K) / K
    prev_ll = -np.inf
    resp = np.ones((n, K)) / K
    converged = False
    for iteration in range(1, int(max_iter) + 1):
        log_joint = np.empty((n, K))
        for c in range(K):
            log_joint[:, c] = math.log(max(pi[c], 1e-300)) + np.array([_pl_logprob_order(o, theta[c]) for o in orders])
        norm = logsumexp(log_joint, axis=1)
        ll = float(norm.sum())
        resp = np.exp(log_joint - norm[:, None])
        pi = np.clip(resp.mean(axis=0), 1e-8, None); pi /= pi.sum()
        for c in range(K):
            fit = _fit_pl_component(orders, resp[:, c], theta[c])
            theta[c] = fit.x
        if abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            converged = True
            break
        prev_ll = ll
    # final likelihood after M-step
    log_joint = np.empty((n, K))
    for c in range(K):
        log_joint[:, c] = math.log(max(pi[c], 1e-300)) + np.array([_pl_logprob_order(o, theta[c]) for o in orders])
    norm = logsumexp(log_joint, axis=1)
    ll = float(norm.sum()); resp = np.exp(log_joint - norm[:, None])
    worth_rows = []
    for c in range(K):
        w = np.exp(np.r_[theta[c], 0.0]); w /= w.sum()
        for item, val in zip(rank_columns, w):
            worth_rows.append({"class": c+1, "class_weight": pi[c], "item": item, "worth": val})
    assignment = pd.DataFrame({"row_index": data.index, "modal_class": resp.argmax(axis=1)+1, "max_posterior": resp.max(axis=1)})
    for c in range(K):
        assignment[f"P(class_{c+1})"] = resp[:, c]
    params = K * (k_items - 1) + (K - 1)
    fit_table = pd.DataFrame([{"n_rankings": n, "classes": K, "items": k_items, "log_likelihood": ll, "aic": 2*params-2*ll, "bic": params*np.log(n)-2*ll, "converged": converged, "iterations": iteration}])
    warns = [] if converged else ["Plackett–Luce mixture EM reached the iteration limit before the requested tolerance."]
    return AnalysisResult({"Class worths": pd.DataFrame(worth_rows), "Membership": assignment, "Fit": fit_table}, pd.DataFrame(), {"method": "Plackett-Luce mixture", "components": K, "seed": seed}, warns)


def plackett_luce_model_selection(
    df: pd.DataFrame,
    *,
    rank_columns: Sequence[str],
    max_components: int = 5,
    seed: int = 42,
    n_init: int = 5,
    criterion: str = "aic",
) -> AnalysisResult:
    """Compare one-component Plackett-Luce with finite mixtures.

    K=1 uses the ordinary Plackett-Luce fit.  For K>=2 several deterministic
    seed offsets are tried and the highest-likelihood solution is retained
    before AIC/BIC model comparison.
    """
    criterion = str(criterion).lower()
    if criterion not in {"aic", "bic"}:
        raise ValueError("Plackett-Luce model selection criterion must be 'aic' or 'bic'.")
    hi = int(max_components)
    if hi < 1 or hi > 5:
        raise ValueError("Plackett-Luce model selection supports 1–5 components.")
    candidates: dict[int, AnalysisResult] = {}
    one = plackett_luce(df, rank_columns=rank_columns)
    candidates[1] = one
    rows = []
    r = one.tables["Fit"].iloc[0].to_dict(); r["candidate_classes"] = 1; r["best_seed"] = np.nan; rows.append(r)
    for K in range(2, hi + 1):
        best = None
        best_seed = None
        for start in range(max(1, int(n_init))):
            run_seed = int(seed) + 1009 * start + 7919 * K
            cand = plackett_luce_mixture(df, rank_columns=rank_columns, components=K, seed=run_seed)
            ll = float(cand.tables["Fit"].iloc[0]["log_likelihood"])
            if best is None or ll > float(best.tables["Fit"].iloc[0]["log_likelihood"]):
                best = cand; best_seed = run_seed
        assert best is not None
        candidates[K] = best
        r = best.tables["Fit"].iloc[0].to_dict(); r["candidate_classes"] = K; r["best_seed"] = best_seed; rows.append(r)
    comparison = pd.DataFrame(rows).sort_values("candidate_classes").reset_index(drop=True)
    selected_idx = pd.to_numeric(comparison[criterion], errors="coerce").idxmin()
    selected_k = int(comparison.loc[selected_idx, "candidate_classes"])
    comparison["selected"] = comparison["candidate_classes"].eq(selected_k)
    selected = candidates[selected_k]
    tables = {"Model comparison": comparison}
    for name, table in selected.tables.items():
        tables[f"Selected K={selected_k} - {name}"] = table
    diagnostics = pd.DataFrame([
        {"diagnostic": "Selection criterion", "value": criterion.upper(), "detail": f"Lowest {criterion.upper()} selected K={selected_k}."},
        {"diagnostic": "Mixture starts", "value": int(n_init), "detail": f"K>=2 evaluated with deterministic seed offsets from base seed {seed}."},
    ])
    warns=[]
    for K, result in candidates.items():
        warns.extend([f"K={K}: {w}" for w in result.warnings])
    return AnalysisResult(tables, diagnostics, {"method":"Plackett-Luce model selection","rank_columns":list(rank_columns),"max_components":hi,"criterion":criterion,"selected_components":selected_k,"seed":int(seed),"n_init":int(n_init)}, warns, selected.raw_result)


# ---------------------------------------------------------------------------
# 6) MCA + Ward and latent-class analysis
# ---------------------------------------------------------------------------

def mca_ward(
    df: pd.DataFrame,
    *, categorical_columns: Sequence[str],
    dimensions: int = 5,
    clusters: int = 3,
    ward_dimensions: int | None = 2,
    benzecri: bool = True,
) -> AnalysisResult:
    """Multiple correspondence analysis followed by Ward clustering.

    Raw MCA eigenvalues are always retained.  When ``benzecri`` is true the
    output also reports Benzécri-corrected inertias, which are commonly used to
    interpret MCA dimensional importance when many indicator columns inflate
    raw inertia.  Ward clustering can deliberately use fewer dimensions than
    are exported for inspection.
    """
    categorical_columns = list(dict.fromkeys(categorical_columns))
    work = df[categorical_columns].astype("string").fillna("<MISSING>")
    if len(work) < 5 or len(categorical_columns) < 2:
        raise ValueError("MCA requires at least five rows and two categorical variables.")
    G = pd.get_dummies(work, prefix=categorical_columns, prefix_sep="=", dtype=float)
    X = G.to_numpy(float)
    total = X.sum()
    P = X / total
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    keep_c = c > 0
    P = P[:, keep_c]; c = c[keep_c]
    G = G.loc[:, keep_c]
    expected = r[:, None] * c[None, :]
    S = (P - expected) / np.sqrt(np.clip(r[:, None] * c[None, :], 1e-300, None))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    eig = s**2
    max_dims = max(1, len(s))
    d = min(max(int(dimensions), 1), max_dims)
    wd = d if ward_dimensions is None else min(max(int(ward_dimensions), 1), d)
    row_coords = (U[:, :d] * s[:d]) / np.sqrt(np.clip(r[:, None], 1e-300, None))
    col_coords = (Vt[:d].T * s[:d]) / np.sqrt(np.clip(c[:, None], 1e-300, None))
    row_df = pd.DataFrame(row_coords, index=work.index, columns=[f"Dim{i+1}" for i in range(d)]).reset_index(names="row_index")
    col_df = pd.DataFrame(col_coords, index=G.columns, columns=[f"Dim{i+1}" for i in range(d)]).reset_index(names="category")

    raw_pct = 100 * eig / eig.sum() if eig.sum() > 0 else np.full_like(eig, np.nan)
    q = len(categorical_columns)
    threshold = 1.0 / q
    corrected = np.where(eig > threshold, (q / (q - 1.0))**2 * np.square(eig - threshold), 0.0)
    corr_sum = corrected.sum()
    corr_pct = 100 * corrected / corr_sum if corr_sum > 0 else np.zeros_like(corrected)
    inertia = pd.DataFrame({
        "dimension": np.arange(1, len(eig)+1),
        "eigenvalue_raw": eig,
        "inertia_pct_raw": raw_pct,
        "cumulative_pct_raw": np.cumsum(raw_pct),
        "benzecri_threshold_1_over_Q": threshold,
        "eigenvalue_benzecri": corrected if benzecri else np.nan,
        "inertia_pct_benzecri": corr_pct if benzecri else np.nan,
        "cumulative_pct_benzecri": np.cumsum(corr_pct) if benzecri else np.nan,
    })

    cluster_coords = row_coords[:, :wd]
    z = linkage(cluster_coords, method="ward")
    cl = fcluster(z, t=int(clusters), criterion="maxclust")
    assignments = pd.DataFrame({"row_index": work.index, "cluster": cl})
    profiles = pd.concat([work.reset_index(names="row_index"), assignments.drop(columns="row_index")], axis=1)
    profile_rows = []
    for cluster_id, g in profiles.groupby("cluster"):
        for col in categorical_columns:
            counts = g[col].value_counts(normalize=True)
            for level, prop in counts.items():
                profile_rows.append({"cluster": cluster_id, "variable": col, "level": str(level), "proportion": prop, "n_cluster": len(g)})
    cluster_sizes = assignments["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="n")
    diagnostics = pd.DataFrame([
        {"diagnostic": "MCA variables", "value": q, "detail": f"Benzécri threshold = 1/Q = {threshold:.8g}."},
        {"diagnostic": "Ward dimensions", "value": wd, "detail": f"Ward linkage uses the first {wd} exported MCA dimensions."},
        {"diagnostic": "Benzécri correction", "value": bool(benzecri), "detail": "Raw eigenvalues are always retained alongside corrected inertia."},
    ])
    return AnalysisResult(
        {
            "Row coordinates": row_df, "Category coordinates": col_df,
            "Inertia": inertia, "Ward assignments": assignments,
            "Cluster sizes": cluster_sizes,
            "Cluster category profiles": pd.DataFrame(profile_rows),
        },
        diagnostics,
        {
            "method": "MCA + Ward", "columns": categorical_columns,
            "dimensions_exported": d, "ward_dimensions": wd,
            "clusters": int(clusters), "benzecri": bool(benzecri),
        },
        [],
    )

def latent_class_analysis(
    df: pd.DataFrame,
    *, categorical_columns: Sequence[str],
    classes: int = 3,
    seed: int = 42,
    n_init: int = 5,
    max_iter: int = 500,
    tol: float = 1e-7,
    smoothing: float = 1e-4,
) -> AnalysisResult:
    work = df[list(categorical_columns)].astype("string").fillna("<MISSING>")
    n = len(work); K = int(classes)
    if K < 2 or K > 8:
        raise ValueError("Interactive LCA supports 2–8 latent classes.")
    if n < max(30, 8*K):
        raise ValueError("Too few observations for the requested latent-class model.")
    codes = []
    levels: list[list[str]] = []
    for col in categorical_columns:
        cats = sorted(work[col].astype(str).unique().tolist())
        mapper = {v: i for i, v in enumerate(cats)}
        codes.append(work[col].astype(str).map(mapper).to_numpy(int))
        levels.append(cats)
    X = np.column_stack(codes)
    rng = np.random.default_rng(seed)
    best = None
    for init in range(int(n_init)):
        resp = rng.dirichlet(np.ones(K), size=n)
        pi = resp.mean(axis=0)
        probs = []
        for j, lev in enumerate(levels):
            q = len(lev)
            Pj = np.empty((K, q))
            for c in range(K):
                counts = np.bincount(X[:, j], weights=resp[:, c], minlength=q) + smoothing
                Pj[c] = counts / counts.sum()
            probs.append(Pj)
        prev = -np.inf
        converged = False
        for iteration in range(1, int(max_iter)+1):
            log_joint = np.tile(np.log(np.clip(pi, 1e-300, None)), (n,1))
            for j, Pj in enumerate(probs):
                for c in range(K):
                    log_joint[:, c] += np.log(np.clip(Pj[c, X[:, j]], 1e-300, None))
            norm = logsumexp(log_joint, axis=1)
            ll = float(norm.sum())
            resp = np.exp(log_joint - norm[:, None])
            pi = np.clip(resp.mean(axis=0), 1e-12, None); pi /= pi.sum()
            for j, lev in enumerate(levels):
                q = len(lev); Pj = np.empty((K,q))
                for c in range(K):
                    counts = np.bincount(X[:, j], weights=resp[:, c], minlength=q) + smoothing
                    Pj[c] = counts/counts.sum()
                probs[j] = Pj
            if abs(ll-prev) < tol*(1+abs(prev)):
                converged=True; break
            prev=ll
        candidate=(ll, resp.copy(), pi.copy(), [p.copy() for p in probs], converged, iteration)
        if best is None or candidate[0] > best[0]:
            best=candidate
    assert best is not None
    ll, resp, pi, probs, converged, iterations = best
    params = (K-1) + sum(K*(len(lev)-1) for lev in levels)
    membership = pd.DataFrame({"row_index": work.index, "modal_class": resp.argmax(axis=1)+1, "max_posterior": resp.max(axis=1)})
    for c in range(K): membership[f"P(class_{c+1})"] = resp[:, c]
    profile_rows=[]
    for j, col in enumerate(categorical_columns):
        for c in range(K):
            for level, prob in zip(levels[j], probs[j][c]):
                profile_rows.append({"class": c+1, "class_weight": pi[c], "variable": col, "level": level, "conditional_probability": prob})
    entropy = -float(np.sum(resp*np.log(np.clip(resp,1e-300,None))))
    fit = pd.DataFrame([{"n":n,"classes":K,"parameters":params,"log_likelihood":ll,"aic":2*params-2*ll,"bic":params*np.log(n)-2*ll,"classification_entropy":entropy,"normalised_entropy":entropy/(n*np.log(K)),"converged":converged,"iterations":iterations}])
    warns=[] if converged else ["Best LCA start reached the iteration limit before tolerance."]
    return AnalysisResult({"Class profiles":pd.DataFrame(profile_rows),"Membership":membership,"Fit":fit},pd.DataFrame(),{"method":"Latent class analysis","columns":list(categorical_columns),"classes":K,"seed":seed,"n_init":n_init},warns)


def latent_class_model_selection(
    df: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    min_classes: int = 2,
    max_classes: int = 5,
    seed: int = 42,
    n_init: int = 10,
    criterion: str = "bic",
) -> AnalysisResult:
    """Fit an LCA class-count sweep and return the selected model.

    The same variables, seed and number of random starts are used for every K.
    BIC is the default selection rule because it penalises latent-class
    proliferation more strongly than AIC, but AIC can be selected explicitly.
    """
    criterion = str(criterion).lower()
    if criterion not in {"bic", "aic"}:
        raise ValueError("LCA model selection criterion must be 'bic' or 'aic'.")
    lo, hi = int(min_classes), int(max_classes)
    if lo < 2 or hi > 8 or lo > hi:
        raise ValueError("LCA class sweep must satisfy 2 <= min_classes <= max_classes <= 8.")
    results: dict[int, AnalysisResult] = {}
    rows: list[dict[str, Any]] = []
    for K in range(lo, hi + 1):
        result = latent_class_analysis(
            df, categorical_columns=categorical_columns, classes=K,
            seed=int(seed), n_init=int(n_init),
        )
        results[K] = result
        fit_row = result.tables["Fit"].iloc[0].to_dict()
        fit_row["candidate_classes"] = K
        rows.append(fit_row)
    comparison = pd.DataFrame(rows).sort_values("candidate_classes").reset_index(drop=True)
    score_col = criterion
    valid = comparison[np.isfinite(pd.to_numeric(comparison[score_col], errors="coerce"))]
    if valid.empty:
        raise ValueError("No finite LCA information criterion was produced.")
    selected_k = int(valid.loc[pd.to_numeric(valid[score_col], errors="coerce").idxmin(), "candidate_classes"])
    comparison["selected"] = comparison["candidate_classes"].eq(selected_k)
    selected = results[selected_k]
    tables = {"Model comparison": comparison}
    for name, table in selected.tables.items():
        tables[f"Selected K={selected_k} - {name}"] = table
    diagnostics = pd.DataFrame([
        {"diagnostic": "Selection criterion", "value": criterion.upper(), "detail": f"Lowest {criterion.upper()} selected K={selected_k}."},
        {"diagnostic": "Candidate range", "value": f"{lo}–{hi}", "detail": f"{n_init} random starts per candidate; seed={seed}."},
    ])
    warns = []
    for K, result in results.items():
        warns.extend([f"K={K}: {w}" for w in result.warnings])
    return AnalysisResult(
        tables, diagnostics,
        {
            "method": "Latent class model selection",
            "columns": list(categorical_columns), "min_classes": lo,
            "max_classes": hi, "criterion": criterion,
            "selected_classes": selected_k, "seed": int(seed),
            "n_init": int(n_init),
        },
        warns, selected.raw_result,
    )


# ---------------------------------------------------------------------------
# 7) Rare but broadly useful inferential utilities
# ---------------------------------------------------------------------------

def dunn_posthoc(
    df: pd.DataFrame,
    *, value: str,
    group: str,
    adjustment: str = "holm",
) -> AnalysisResult:
    work = pd.concat([pd.to_numeric(df[value], errors="coerce").rename("value"), df[group].astype("string").rename("group")], axis=1).dropna()
    groups = sorted(work.group.unique().tolist())
    if len(groups) < 2:
        raise ValueError("Dunn post-hoc requires at least two groups.")
    ranks = stats.rankdata(work.value.to_numpy(float), method="average")
    work = work.assign(rank=ranks)
    N = len(work)
    _, tie_counts = np.unique(work.value.to_numpy(float), return_counts=True)
    tie_correction = 1.0 - np.sum(tie_counts**3 - tie_counts) / (N**3 - N) if N > 1 else 1.0
    var_rank = N*(N+1)/12.0 * tie_correction
    rows=[]
    for a,b in combinations(groups,2):
        ga=work[work.group==a]; gb=work[work.group==b]
        diff=ga["rank"].mean()-gb["rank"].mean()
        se=math.sqrt(max(var_rank*(1/len(ga)+1/len(gb)),1e-300))
        z=diff/se; p=2*stats.norm.sf(abs(z))
        rows.append({"group_a":a,"group_b":b,"mean_rank_difference":diff,"z":z,"p_value":p,"n_a":len(ga),"n_b":len(gb)})
    out=pd.DataFrame(rows)
    out["p_adjusted"]=multipletests(out.p_value,method=adjustment)[1]
    return AnalysisResult({"Dunn pairwise":out},pd.DataFrame(),{"method":"Dunn post-hoc","value":value,"group":group,"adjustment":adjustment},[])


def equivalence_tost(
    df: pd.DataFrame,
    *, value_a: str,
    value_b: str | None = None,
    group: str | None = None,
    low: float,
    high: float,
    paired: bool = True,
) -> AnalysisResult:
    if low >= high:
        raise ValueError("TOST lower equivalence bound must be below the upper bound.")
    if paired:
        if value_b is None:
            raise ValueError("Paired TOST requires value_b.")
        work = _as_numeric_frame(df,[value_a,value_b]).dropna()
        result = ttost_paired(work[value_a],work[value_b],low,high)
        n=len(work)
    else:
        if group is None:
            raise ValueError("Independent TOST requires a group column with exactly two observed groups.")
        work=pd.concat([pd.to_numeric(df[value_a],errors="coerce").rename("value"),df[group].rename("group")],axis=1).dropna()
        lev=work.group.unique()
        if len(lev)!=2: raise ValueError("Independent TOST requires exactly two groups.")
        a=work.loc[work.group==lev[0],"value"]; b=work.loc[work.group==lev[1],"value"]
        result=ttost_ind(a,b,low,high,usevar="unequal")
        n=len(work)
    p_equiv=float(result[0])
    table=pd.DataFrame([{"n":n,"lower_bound":low,"upper_bound":high,"TOST_p_value":p_equiv,"equivalent_at_5pct":p_equiv<.05}])
    return AnalysisResult({"Equivalence test":table},pd.DataFrame(),{"method":"TOST","paired":paired,"low":low,"high":high},[])


def meta_analysis(
    df: pd.DataFrame,
    *, effect: str,
    standard_error: str,
    study_label: str | None = None,
) -> AnalysisResult:
    from statsmodels.stats.meta_analysis import combine_effects
    eff=pd.to_numeric(df[effect],errors="coerce")
    se=pd.to_numeric(df[standard_error],errors="coerce")
    work=pd.DataFrame({"effect":eff,"variance":se**2})
    if study_label: work["study"]=df[study_label].astype(str)
    work=work.replace([np.inf,-np.inf],np.nan).dropna()
    if (work.variance<=0).any() or len(work)<2: raise ValueError("Meta-analysis requires at least two studies with positive standard errors.")
    res=combine_effects(work.effect.to_numpy(),work.variance.to_numpy(),method_re="iterated",use_t=False)
    # statsmodels exposes fixed/random summaries as attributes.
    summary=res.summary_frame()
    summary=summary.reset_index(names="row")
    fit=pd.DataFrame([{"studies":len(work),"tau_squared":float(res.tau2),"Q":float(res.q),"Q_df":int(res.k-1),"I_squared":float(res.i2),"H_squared":float(res.h2)}])
    return AnalysisResult({"Meta-analysis summary":summary,"Heterogeneity":fit,"Study inputs":work.reset_index(names="row_index")},pd.DataFrame(),{"method":"Fixed/random-effects meta-analysis","effect":effect,"standard_error":standard_error},[],res)


def rasch_1pl(
    df: pd.DataFrame,
    *, item_columns: Sequence[str],
    max_iter: int = 1000,
) -> AnalysisResult:
    """Binary Rasch 1PL model by joint maximum likelihood with identifiability centring.

    Intended as a rare diagnostic/survey module, not a replacement for a full
    IRT package with marginal ML, DIF and complex sampling support.
    """
    data=_as_numeric_frame(df,item_columns).dropna()
    X=data.to_numpy(float)
    if not np.all(np.isin(X,[0,1])): raise ValueError("Rasch 1PL requires binary 0/1 item responses.")
    n,j=X.shape
    if n<20 or j<3: raise ValueError("Rasch 1PL requires at least 20 complete respondents and 3 items.")
    # theta persons and beta item difficulties, with beta[-1] implied by centring.
    def unpack(par):
        theta=par[:n]
        beta_free=par[n:]
        beta=np.r_[beta_free,-beta_free.sum()]
        theta=theta-theta.mean()
        return theta,beta
    def nll(par):
        theta,beta=unpack(par)
        eta=theta[:,None]-beta[None,:]
        return float(np.sum(np.logaddexp(0,eta)-X*eta))
    start=np.zeros(n+j-1)
    res=optimize.minimize(nll,start,method="L-BFGS-B",options={"maxiter":int(max_iter),"ftol":1e-10})
    theta,beta=unpack(res.x)
    item=pd.DataFrame({"item":item_columns,"difficulty":beta}).sort_values("difficulty")
    person=pd.DataFrame({"row_index":data.index,"ability":theta})
    fit=pd.DataFrame([{"n_persons":n,"items":j,"log_likelihood":-float(res.fun),"aic":2*(n+j-1)+2*float(res.fun),"converged":bool(res.success)}])
    warns=["Joint-ML Rasch ability estimates are biased in short tests; use this module as a rare diagnostic and prefer dedicated marginal-ML IRT for definitive psychometrics."]
    return AnalysisResult({"Item difficulties":item,"Person abilities":person,"Fit":fit},pd.DataFrame(),{"method":"Rasch 1PL joint ML","items":list(item_columns)},warns)


def regression_discontinuity(
    df: pd.DataFrame,
    *, y: str,
    running: str,
    cutoff: float,
    bandwidth: float,
    covariates: Sequence[str] = (),
    kernel: str = "triangular",
) -> AnalysisResult:
    if bandwidth<=0: raise ValueError("RDD bandwidth must be positive.")
    cols=[y,running,*covariates]
    work=_as_numeric_frame(df,cols).replace([np.inf,-np.inf],np.nan).dropna()
    x=work[running]-float(cutoff)
    keep=x.abs()<=float(bandwidth)
    work=work.loc[keep].copy(); x=x.loc[keep]
    if len(work)<30: raise ValueError("Fewer than 30 complete observations remain inside the requested RDD bandwidth.")
    treat=(x>=0).astype(float)
    X=pd.DataFrame({"const":1.0,"running_centered":x,"treatment":treat,"treatment_x_running":treat*x},index=work.index)
    for c in covariates: X[c]=work[c]
    u=(x.abs()/float(bandwidth)).to_numpy(float)
    weights=(1-u) if kernel=="triangular" else np.ones(len(work))
    weights=np.clip(weights,1e-8,None)
    result=sm.WLS(work[y],X,weights=weights).fit(cov_type="HC1")
    coef=_coef_table(result)
    effect=coef.loc[coef.term=="treatment"].copy()
    fit=pd.DataFrame([{"n_bandwidth":len(work),"cutoff":cutoff,"bandwidth":bandwidth,"kernel":kernel,"left_n":int((x<0).sum()),"right_n":int((x>=0).sum()),"r_squared":float(result.rsquared)}])
    diagnostics=pd.DataFrame([{"diagnostic":"Identification","value":"Local continuity assumption required","detail":"Inspect manipulation around the cutoff, covariate balance and bandwidth sensitivity before causal interpretation."}])
    return AnalysisResult({"RDD effect":effect,"All coefficients":coef,"Fit":fit},diagnostics,{"method":"Local linear RDD","outcome":y,"running":running,"cutoff":cutoff,"bandwidth":bandwidth,"kernel":kernel},[],result)


METHOD_CATALOGUE = pd.DataFrame([
    ["Firth logistic", "Binary / rare events", "Separation or small-event bias", "Confirmatory", "Very uncommon but decisive when ordinary logit fails"],
    ["Ordered logit/probit", "Ordinal outcome", "3+ ordered levels", "Confirmatory", "Common in survey/policy research"],
    ["Brant-type Wald", "Ordinal diagnostics", "Check proportional-odds assumption", "Diagnostic", "Specialist"],
    ["Multinomial logit", "Nominal outcome", "3+ unordered levels", "Confirmatory", "Specialist"],
    ["Beta regression", "Bounded continuous", "Outcome strictly in (0,1)", "Confirmatory", "Specialist"],
    ["Tobit", "Censored continuous", "Known lower/upper censoring", "Confirmatory", "Rare"],
    ["ZIP / ZINB", "Zero-inflated count", "Excess zeros", "Confirmatory", "Specialist"],
    ["Linear mixed effects", "Repeated/hierarchical", "Random intercept/slope", "Confirmatory", "Common in multilevel research"],
    ["GEE", "Clustered/repeated", "Population-average effects", "Confirmatory", "Specialist"],
    ["Cox PH", "Time-to-event", "Right-censored survival", "Confirmatory", "Specialist"],
    ["CLR / ILR", "Compositional", "Parts sum to a constant", "Transform", "Specialist"],
    ["PERMANOVA on ILR", "Compositional/multivariate", "Group difference in composition", "Confirmatory", "Rare"],
    ["Dirichlet regression", "Compositional", "Composition conditional on predictors", "Confirmatory", "Rare"],
    ["Friedman + Wilcoxon/Holm", "Repeated rankings/weights", "Within-respondent comparisons", "Confirmatory", "Common non-parametric"],
    ["Plackett-Luce", "Complete rankings", "Latent item worth", "Confirmatory", "Rare"],
    ["Plackett-Luce mixture", "Ranking heterogeneity", "Latent ranking classes", "Latent", "Very rare"],
    ["MCA + Ward", "Categorical multivariate", "Latent categorical geometry", "Exploratory", "Specialist"],
    ["Latent class analysis", "Categorical multivariate", "Conditional-independence classes", "Latent", "Rare"],
    ["Dunn post-hoc", "Non-parametric groups", "After Kruskal-Wallis", "Post-hoc", "Common specialist"],
    ["TOST equivalence", "Means / differences", "Demonstrate practical equivalence", "Confirmatory", "Rare but important"],
    ["Meta-analysis", "Study-level effects", "Fixed/random pooling", "Synthesis", "Specialist"],
    ["Rasch 1PL", "Binary item responses", "Item difficulty/person ability", "Psychometrics", "Very rare in general workbench"],
    ["Local linear RDD", "Quasi-experimental", "Known assignment cutoff", "Causal", "Rare and assumption-heavy"],
], columns=["method","data_family","use_when","role","rarity"])

# ---------------------------------------------------------------------------
# 8) Ultra-rare exact, matched-design and quasi-experimental utilities
# ---------------------------------------------------------------------------

def conditional_logistic(
    df: pd.DataFrame,
    *, y: str,
    x_vars: Sequence[str],
    strata: str,
) -> AnalysisResult:
    """Conditional logistic regression for matched/stratified binary outcomes."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    yy = pd.to_numeric(df[y], errors="coerce")
    X = _design_matrix(df, x_vars, (), add_constant=False)
    joined = pd.concat([yy.rename("__y__"), df[strata].rename("__strata__"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if not set(np.unique(joined["__y__"])).issubset({0,1}):
        raise ValueError("Conditional logistic regression requires a binary 0/1 outcome.")
    # Strata with no within-stratum outcome variation carry no conditional likelihood information.
    variation = joined.groupby("__strata__")["__y__"].nunique()
    informative = variation[variation > 1].index
    dropped = int((~joined["__strata__"].isin(informative)).sum())
    work = joined[joined["__strata__"].isin(informative)].copy()
    if work["__strata__"].nunique() < 2:
        raise ValueError("Fewer than two informative strata remain after removing strata with no outcome variation.")
    model = ConditionalLogit(work["__y__"].astype(int), work[X.columns], groups=work["__strata__"])
    result = model.fit(method="bfgs", maxiter=1000, disp=False)
    coef = _coef_table(result, exponentiate=True, effect_name="odds_ratio")
    fit = pd.DataFrame([{"n": len(work), "informative_strata": int(work["__strata__"].nunique()), "dropped_rows_uninformative_strata": dropped, "log_likelihood": float(result.llf)}])
    return AnalysisResult({"Coefficients": coef, "Fit": fit}, pd.DataFrame(), {"method":"Conditional logistic","outcome":y,"predictors":list(x_vars),"strata":strata}, [], result)


def exact_2x2_tests(
    df: pd.DataFrame,
    *, outcome: str,
    exposure: str,
) -> AnalysisResult:
    """Fisher, Barnard and Boschloo exact tests for a 2x2 table."""
    y = pd.to_numeric(df[outcome], errors="coerce")
    x = pd.to_numeric(df[exposure], errors="coerce")
    work = pd.DataFrame({"y":y,"x":x}).dropna()
    if not set(np.unique(work.y)).issubset({0,1}) or not set(np.unique(work.x)).issubset({0,1}):
        raise ValueError("Exact 2x2 tests require binary 0/1 outcome and exposure.")
    table = pd.crosstab(work.x.astype(int), work.y.astype(int)).reindex(index=[0,1], columns=[0,1], fill_value=0)
    arr = table.to_numpy(int)
    fisher = stats.fisher_exact(arr, alternative="two-sided")
    barnard = stats.barnard_exact(arr, alternative="two-sided")
    boschloo = stats.boschloo_exact(arr, alternative="two-sided")
    a,b = arr[1,1],arr[1,0]
    c,d = arr[0,1],arr[0,0]
    aa,bb,cc,dd = [v + 0.5 if 0 in arr else v for v in (a,b,c,d)]
    or_hat = aa*dd/(bb*cc)
    se_log_or = math.sqrt(1/aa+1/bb+1/cc+1/dd)
    rr = (aa/(aa+bb))/(cc/(cc+dd))
    tests = pd.DataFrame([
        {"test":"Fisher exact","statistic_or_odds_ratio":float(fisher.statistic),"p_value":float(fisher.pvalue)},
        {"test":"Barnard exact","statistic_or_odds_ratio":float(barnard.statistic),"p_value":float(barnard.pvalue)},
        {"test":"Boschloo exact","statistic_or_odds_ratio":float(boschloo.statistic),"p_value":float(boschloo.pvalue)},
    ])
    effects = pd.DataFrame([{"odds_ratio_haldane_if_needed":or_hat,"or_ci_95_low":math.exp(math.log(or_hat)-1.95996398454*se_log_or),"or_ci_95_high":math.exp(math.log(or_hat)+1.95996398454*se_log_or),"risk_ratio_haldane_if_needed":rr,"continuity_correction_used":bool((arr==0).any())}])
    counts = table.reset_index(names="exposure")
    return AnalysisResult({"2x2 counts":counts,"Exact tests":tests,"Effect sizes":effects}, pd.DataFrame(), {"method":"Exact 2x2 suite","outcome":outcome,"exposure":exposure}, [])


def mantel_haenszel(
    df: pd.DataFrame,
    *, outcome: str,
    exposure: str,
    strata: str,
) -> AnalysisResult:
    from statsmodels.stats.contingency_tables import StratifiedTable
    y = pd.to_numeric(df[outcome], errors="coerce")
    x = pd.to_numeric(df[exposure], errors="coerce")
    work = pd.DataFrame({"y":y,"x":x,"s":df[strata]}).dropna()
    if not set(np.unique(work.y)).issubset({0,1}) or not set(np.unique(work.x)).issubset({0,1}):
        raise ValueError("Mantel-Haenszel requires binary 0/1 outcome and exposure.")
    arrays=[]; used=[]
    for s,g in work.groupby("s"):
        tab=pd.crosstab(g.x.astype(int),g.y.astype(int)).reindex(index=[0,1],columns=[0,1],fill_value=0).to_numpy(float)
        if tab.sum()>0:
            arrays.append(tab); used.append(s)
    if len(arrays)<2: raise ValueError("At least two non-empty strata are required.")
    cube=np.stack(arrays,axis=2)
    res=StratifiedTable(cube)
    test=res.test_null_odds()
    hom=res.test_equal_odds()
    summary=pd.DataFrame([{"strata":len(arrays),"common_odds_ratio":float(res.oddsratio_pooled),"or_ci_95_low":float(res.oddsratio_pooled_confint()[0]),"or_ci_95_high":float(res.oddsratio_pooled_confint()[1]),"CMH_chi_square":float(test.statistic),"CMH_p_value":float(test.pvalue),"Breslow_Day_chi_square":float(hom.statistic),"Breslow_Day_p_value":float(hom.pvalue)}])
    return AnalysisResult({"Mantel-Haenszel":summary,"Strata used":pd.DataFrame({"stratum":used})},pd.DataFrame(),{"method":"Mantel-Haenszel","outcome":outcome,"exposure":exposure,"strata":strata},[],res)


def page_trend(
    df: pd.DataFrame,
    *, columns: Sequence[str],
    ranked: bool = False,
) -> AnalysisResult:
    data=_as_numeric_frame(df,columns).dropna()
    if len(data)<5 or len(columns)<3: raise ValueError("Page trend test requires at least five complete blocks and three ordered conditions.")
    res=stats.page_trend_test(data.to_numpy(float), ranked=ranked)
    table=pd.DataFrame([{"n_blocks":len(data),"ordered_conditions":len(columns),"Page_L":float(res.statistic),"p_value":float(res.pvalue),"method":str(res.method)}])
    return AnalysisResult({"Page trend test":table},pd.DataFrame(),{"method":"Page L trend test","columns":list(columns),"ranked_input":ranked},[])


def alexander_govern_test(
    df: pd.DataFrame,
    *, value: str,
    group: str,
) -> AnalysisResult:
    work=pd.DataFrame({"value":pd.to_numeric(df[value],errors="coerce"),"group":df[group]}).dropna()
    samples=[g.value.to_numpy(float) for _,g in work.groupby("group") if len(g)>=2]
    if len(samples)<2: raise ValueError("Alexander-Govern requires at least two groups with at least two observations each.")
    res=stats.alexandergovern(*samples)
    table=pd.DataFrame([{"groups":len(samples),"n":sum(map(len,samples)),"Alexander_Govern_statistic":float(res.statistic),"p_value":float(res.pvalue)}])
    return AnalysisResult({"Alexander-Govern":table},pd.DataFrame(),{"method":"Alexander-Govern","value":value,"group":group},[])


def heckman_two_step(
    df: pd.DataFrame,
    *, y: str,
    selection: str,
    outcome_predictors: Sequence[str],
    selection_predictors: Sequence[str],
    categorical: Sequence[str] = (),
) -> AnalysisResult:
    """Heckman two-step selection correction (probit + inverse Mills ratio)."""
    sel=pd.to_numeric(df[selection],errors="coerce")
    Xs=_design_matrix(df,selection_predictors,categorical,add_constant=True)
    sel_join=pd.concat([sel.rename("__sel__"),Xs],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if not set(np.unique(sel_join.__sel__)).issubset({0,1}): raise ValueError("Selection indicator must be binary 0/1.")
    probit=sm.Probit(sel_join.__sel__,sel_join[Xs.columns]).fit(disp=False,maxiter=500)
    z=pd.Series(probit.predict(which="linear"),index=sel_join.index)
    phi=stats.norm.pdf(z)
    Phi=np.clip(stats.norm.cdf(z),1e-12,1-1e-12)
    imr=pd.Series(phi/Phi,index=sel_join.index,name="inverse_mills_ratio")
    Xo=_design_matrix(df,outcome_predictors,categorical,add_constant=True)
    yy=pd.to_numeric(df[y],errors="coerce")
    second=pd.concat([yy.rename("__y__"),sel.rename("__sel__"),Xo,imr.rename("inverse_mills_ratio")],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    second=second[second.__sel__==1].copy()
    if len(second)<max(20,Xo.shape[1]+5): raise ValueError("Too few selected observations for the outcome equation.")
    outcome=sm.OLS(second.__y__,second[[*Xo.columns,"inverse_mills_ratio"]]).fit(cov_type="HC3")
    first=_coef_table(probit)
    second_coef=_coef_table(outcome)
    fit=pd.DataFrame([{"selection_n":len(sel_join),"selected_n":len(second),"selection_log_likelihood":float(probit.llf),"outcome_r_squared":float(outcome.rsquared),"IMR_p_value":float(outcome.pvalues.get("inverse_mills_ratio",np.nan))}])
    diagnostics=pd.DataFrame([{"diagnostic":"Exclusion restriction","value":"Strongly recommended","detail":"For credible identification, include at least one selection predictor excluded from the outcome equation."}])
    return AnalysisResult({"Selection probit":first,"Outcome equation":second_coef,"Fit":fit},diagnostics,{"method":"Heckman two-step","outcome":y,"selection":selection,"outcome_predictors":list(outcome_predictors),"selection_predictors":list(selection_predictors)},["Two-step standard errors are approximate; use full-information ML when definitive selection-model inference is required."],{"selection":probit,"outcome":outcome})


def synthetic_control(
    df: pd.DataFrame,
    *, unit: str,
    time: str,
    outcome: str,
    treated_unit: Any,
    intervention_time: Any,
) -> AnalysisResult:
    """Basic Abadie-style synthetic control using pre-treatment outcome paths."""
    work=df[[unit,time,outcome]].copy()
    work[outcome]=pd.to_numeric(work[outcome],errors="coerce")
    work=work.dropna()
    pivot=work.pivot_table(index=time,columns=unit,values=outcome,aggfunc="mean").sort_index()
    if treated_unit not in pivot.columns: raise ValueError("Treated unit is not present in the panel.")
    try:
        pre_mask=np.asarray(pivot.index < intervention_time)
        post_mask=np.asarray(pivot.index >= intervention_time)
    except Exception as exc:
        raise ValueError("Time values and intervention_time must be mutually comparable.") from exc
    donors=[c for c in pivot.columns if c!=treated_unit]
    complete_cols=[c for c in donors if pivot.loc[pre_mask,c].notna().all()]
    if len(complete_cols)<2: raise ValueError("At least two donor units with complete pre-intervention outcomes are required.")
    ypre=pivot.loc[pre_mask,treated_unit].to_numpy(float)
    if np.any(~np.isfinite(ypre)): raise ValueError("Treated unit has missing pre-intervention outcomes.")
    Xpre=pivot.loc[pre_mask,complete_cols].to_numpy(float)
    if len(ypre)<3: raise ValueError("At least three pre-intervention periods are required.")
    def obj(w): return float(np.mean((ypre-Xpre@w)**2))
    cons={"type":"eq","fun":lambda w:np.sum(w)-1}
    bounds=[(0,1)]*len(complete_cols)
    res=optimize.minimize(obj,np.ones(len(complete_cols))/len(complete_cols),method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":5000,"ftol":1e-12})
    if not res.success: raise ValueError(f"Synthetic-control optimisation failed: {res.message}")
    w=np.asarray(res.x); synth=pivot[complete_cols].to_numpy(float)@w
    treated=pivot[treated_unit].to_numpy(float)
    gaps=treated-synth
    time_values=pivot.index.to_numpy()
    series=pd.DataFrame({"time":time_values,"treated":treated,"synthetic":synth,"gap":gaps,"period":np.where(post_mask,"post","pre")})
    weights=pd.DataFrame({"donor_unit":complete_cols,"weight":w}).sort_values("weight",ascending=False)
    pre_rmspe=float(np.sqrt(np.nanmean(gaps[pre_mask]**2)))
    post_rmspe=float(np.sqrt(np.nanmean(gaps[post_mask]**2))) if post_mask.any() else np.nan
    fit=pd.DataFrame([{"treated_unit":str(treated_unit),"intervention_time":str(intervention_time),"donors":len(complete_cols),"pre_periods":int(pre_mask.sum()),"post_periods":int(post_mask.sum()),"pre_RMSPE":pre_rmspe,"post_RMSPE":post_rmspe,"post_pre_RMSPE_ratio":post_rmspe/pre_rmspe if pre_rmspe>0 and np.isfinite(post_rmspe) else np.nan,"mean_post_gap":float(np.nanmean(gaps[post_mask])) if post_mask.any() else np.nan}])
    diagnostics=pd.DataFrame([{"diagnostic":"Inference","value":"Placebo/permutation inference not automatic in this basic module","detail":"Causal interpretation requires no concurrent treated-unit shock and a donor pool capable of reproducing the pre-treatment path."}])
    return AnalysisResult({"Donor weights":weights,"Observed vs synthetic":series,"Fit":fit},diagnostics,{"method":"Synthetic control","unit":unit,"time":time,"outcome":outcome,"treated_unit":str(treated_unit),"intervention_time":str(intervention_time)},[],res)


METHOD_CATALOGUE = pd.concat([
    METHOD_CATALOGUE,
    pd.DataFrame([
        ["Conditional logistic", "Matched binary", "Matched case-control / fixed strata", "Confirmatory", "Very rare"],
        ["Fisher/Barnard/Boschloo exact", "2x2 binary", "Tiny or sparse contingency tables", "Exact", "Ultra-rare beyond Fisher"],
        ["Mantel-Haenszel", "Stratified 2x2", "Common adjusted odds ratio", "Confirmatory", "Specialist"],
        ["Page L trend", "Ordered repeated measures", "Monotonic ordered alternative", "Exact/asymptotic rank", "Very rare"],
        ["Alexander-Govern", "Heteroskedastic groups", "Robust k-sample mean comparison", "Confirmatory", "Very rare"],
        ["Heckman two-step", "Sample selection", "Non-randomly observed outcome", "Econometric", "Rare"],
        ["Synthetic control", "Comparative case study", "One treated unit + donor panel", "Causal/quasi-experimental", "Rare"],
    ], columns=METHOD_CATALOGUE.columns),
], ignore_index=True)

# ---------------------------------------------------------------------------
# Ultra-rare nonparametric, dependence and evidence-synthesis utilities
# ---------------------------------------------------------------------------

def brunner_munzel_test(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    alternative: str = "two-sided",
) -> AnalysisResult:
    """Brunner-Munzel stochastic-equality test for two independent samples.

    Unlike the classical Mann-Whitney interpretation, the Brunner-Munzel test
    does not require equal distributional shapes/variances under the null.
    """
    d = df[[value, group]].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna()
    levels = list(pd.Series(d[group]).drop_duplicates())
    if len(levels) != 2:
        raise ValueError("Brunner-Munzel requires exactly two groups.")
    x = d.loc[d[group] == levels[0], value].to_numpy(float)
    y = d.loc[d[group] == levels[1], value].to_numpy(float)
    if min(len(x), len(y)) < 2:
        raise ValueError("Each group requires at least two observations.")
    res = stats.brunnermunzel(x, y, alternative=alternative, distribution="t")
    # U/(n1*n2) estimates P(X>Y)+.5P(X=Y); report both directions explicitly.
    u = float(stats.mannwhitneyu(x, y, alternative="two-sided", method="auto").statistic)
    ps_x_gt_y = u / (len(x) * len(y))
    table = pd.DataFrame([{
        "group_1": str(levels[0]), "n_1": len(x), "group_2": str(levels[1]), "n_2": len(y),
        "W_statistic": float(res.statistic), "p_value": float(res.pvalue),
        "probability_superiority_group1": ps_x_gt_y,
        "probability_superiority_group2": 1.0 - ps_x_gt_y,
        "alternative": alternative,
    }])
    return AnalysisResult({"Brunner-Munzel": table}, pd.DataFrame(),
                          {"value": value, "group": group, "alternative": alternative}, [], res)


def jonckheere_terpstra(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    order: Sequence[Any] | None = None,
    alternative: str = "increasing",
    permutations: int = 1999,
    seed: int = 42,
) -> AnalysisResult:
    """Jonckheere-Terpstra ordered-alternative test with permutation inference.

    The permutation route is deliberately used because ties are common in
    applied survey data and tie-adjusted asymptotic formulae vary by software.
    """
    d = df[[value, group]].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna()
    if order is None:
        levels = list(pd.Series(d[group]).drop_duplicates())
        try:
            levels = sorted(levels)
        except Exception:
            pass
    else:
        levels = list(order)
    if len(levels) < 3:
        raise ValueError("Jonckheere-Terpstra is intended for three or more ordered groups.")
    unknown = set(pd.Series(d[group]).unique()) - set(levels)
    if unknown:
        raise ValueError("The supplied order omits observed groups: " + ", ".join(map(str, unknown)))
    labels = pd.Categorical(d[group], categories=levels, ordered=True).codes
    values = d[value].to_numpy(float)

    def jt_stat(lbl: np.ndarray) -> float:
        total = 0.0
        for i in range(len(levels) - 1):
            a = values[lbl == i]
            for j in range(i + 1, len(levels)):
                b = values[lbl == j]
                if len(a) and len(b):
                    diff = b[:, None] - a[None, :]
                    total += float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
        return total

    observed = jt_stat(labels)
    rng = np.random.default_rng(seed)
    perms = max(99, int(permutations))
    null = np.empty(perms, dtype=float)
    for b in range(perms):
        null[b] = jt_stat(rng.permutation(labels))
    if alternative == "decreasing":
        p = (1 + np.sum(null <= observed)) / (perms + 1)
    elif alternative == "two-sided":
        centre = float(np.mean(null))
        p = (1 + np.sum(np.abs(null - centre) >= abs(observed - centre))) / (perms + 1)
    else:
        p = (1 + np.sum(null >= observed)) / (perms + 1)
    sd = float(np.std(null, ddof=1))
    z = (observed - float(np.mean(null))) / sd if sd > 0 else np.nan
    summary = pd.DataFrame([{
        "JT_statistic": observed, "permutation_p_value": float(p), "permutation_z": z,
        "groups": len(levels), "n": len(d), "permutations": perms,
        "alternative": alternative, "ordered_groups": " < ".join(map(str, levels)),
    }])
    group_summary = d.groupby(group, observed=False)[value].agg(["count", "mean", "median"]).reset_index()
    return AnalysisResult({"Jonckheere-Terpstra": summary, "Group summary": group_summary}, pd.DataFrame(),
                          {"value": value, "group": group, "order": list(map(str, levels)), "seed": seed}, [], null)


def quade_test(df: pd.DataFrame, *, columns: Sequence[str]) -> AnalysisResult:
    """Quade's rank test for complete randomized blocks / repeated measures."""
    cols = list(columns)
    if len(cols) < 3:
        raise ValueError("Quade's test requires at least three repeated conditions.")
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    b, k = d.shape
    if b < 3:
        raise ValueError("At least three complete blocks are required.")
    x = d.to_numpy(float)
    within_ranks = np.vstack([stats.rankdata(row, method="average") for row in x])
    block_ranges = np.ptp(x, axis=1)
    q = stats.rankdata(block_ranges, method="average")
    s = q[:, None] * (within_ranks - (k + 1) / 2.0)
    a = float(np.sum(s**2))
    treatment_sums = s.sum(axis=0)
    big_b = float(np.sum(treatment_sums**2) / b)
    denom = a - big_b
    f_stat = float((b - 1) * big_b / denom) if denom > 0 else np.inf
    df1 = k - 1
    df2 = (b - 1) * (k - 1)
    p = float(stats.f.sf(f_stat, df1, df2)) if np.isfinite(f_stat) else 0.0
    summary = pd.DataFrame([{"F_statistic": f_stat, "df1": df1, "df2": df2, "p_value": p, "blocks": b, "conditions": k}])
    treatment = pd.DataFrame({"condition": cols, "weighted_rank_sum": treatment_sums, "mean": d.mean().to_numpy(), "median": d.median().to_numpy()})
    return AnalysisResult({"Quade test": summary, "Condition summary": treatment}, pd.DataFrame(), {"columns": cols}, [], None)


def cochran_q_test(df: pd.DataFrame, *, columns: Sequence[str]) -> AnalysisResult:
    """Cochran's Q for three or more matched binary conditions."""
    from statsmodels.stats.contingency_tables import cochrans_q
    cols = list(columns)
    if len(cols) < 3:
        raise ValueError("Cochran Q requires at least three matched binary variables.")
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    vals = set(np.unique(d.to_numpy(float)))
    if not vals.issubset({0.0, 1.0}):
        raise ValueError("Cochran Q variables must be coded 0/1.")
    res = cochrans_q(d.to_numpy(int), return_object=True)
    summary = pd.DataFrame([{"Q_statistic": float(res.statistic), "df": len(cols)-1, "p_value": float(res.pvalue), "complete_subjects": len(d), "conditions": len(cols)}])
    rates = pd.DataFrame({"condition": cols, "positive_rate": d.mean().to_numpy(), "positive_n": d.sum().astype(int).to_numpy()})
    return AnalysisResult({"Cochran Q": summary, "Condition rates": rates}, pd.DataFrame(), {"columns": cols}, [], res)


def mcnemar_test(df: pd.DataFrame, *, variable_a: str, variable_b: str, exact: bool = True) -> AnalysisResult:
    """Paired binary McNemar test."""
    from statsmodels.stats.contingency_tables import mcnemar
    d = df[[variable_a, variable_b]].apply(pd.to_numeric, errors="coerce").dropna()
    vals = set(np.unique(d.to_numpy(float)))
    if not vals.issubset({0.0, 1.0}):
        raise ValueError("McNemar variables must be coded 0/1.")
    tab = pd.crosstab(d[variable_a].astype(int), d[variable_b].astype(int)).reindex(index=[0,1], columns=[0,1], fill_value=0)
    res = mcnemar(tab.to_numpy(), exact=bool(exact), correction=not bool(exact))
    summary = pd.DataFrame([{
        "statistic": float(res.statistic), "p_value": float(res.pvalue), "exact": bool(exact),
        "n": len(d), "discordant_a1_b0": int(tab.loc[1,0]), "discordant_a0_b1": int(tab.loc[0,1]),
    }])
    contingency = tab.rename_axis(index=variable_a, columns=variable_b).reset_index()
    return AnalysisResult({"McNemar": summary, "Paired table": contingency}, pd.DataFrame(), {"variable_a": variable_a, "variable_b": variable_b, "exact": exact}, [], res)


def bowker_symmetry_test(df: pd.DataFrame, *, variable_a: str, variable_b: str) -> AnalysisResult:
    """Bowker test of symmetry: multicategory generalisation of McNemar."""
    d = df[[variable_a, variable_b]].dropna()
    levels = list(dict.fromkeys(list(d[variable_a].unique()) + list(d[variable_b].unique())))
    if len(levels) < 3:
        raise ValueError("Bowker symmetry is most useful for three or more paired categories; use McNemar for binary data.")
    tab = pd.crosstab(d[variable_a], d[variable_b]).reindex(index=levels, columns=levels, fill_value=0)
    stat = 0.0; dfree = 0
    components = []
    for i in range(len(levels)-1):
        for j in range(i+1, len(levels)):
            nij, nji = int(tab.iloc[i,j]), int(tab.iloc[j,i])
            den = nij + nji
            if den > 0:
                comp = (nij-nji)**2 / den
                stat += comp; dfree += 1
                components.append({"category_i": str(levels[i]), "category_j": str(levels[j]), "n_ij": nij, "n_ji": nji, "chi_square_component": comp})
    p = float(stats.chi2.sf(stat, dfree)) if dfree else np.nan
    summary = pd.DataFrame([{"chi_square": stat, "df": dfree, "p_value": p, "paired_n": len(d), "categories": len(levels)}])
    return AnalysisResult({"Bowker symmetry": summary, "Pair components": pd.DataFrame(components), "Contingency": tab.reset_index()}, pd.DataFrame(), {"variable_a": variable_a, "variable_b": variable_b}, [], None)


def _double_center_distance(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    dist = cdist(x, x, metric="euclidean")
    return dist - dist.mean(axis=0, keepdims=True) - dist.mean(axis=1, keepdims=True) + dist.mean()


def distance_correlation_test(
    df: pd.DataFrame,
    *,
    x_columns: Sequence[str],
    y_columns: Sequence[str],
    permutations: int = 999,
    seed: int = 42,
) -> AnalysisResult:
    """Distance-correlation test for arbitrary nonlinear multivariate dependence."""
    xs, ys = list(x_columns), list(y_columns)
    if not xs or not ys:
        raise ValueError("Select at least one X and one Y variable.")
    d = df[list(dict.fromkeys(xs+ys))].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 5:
        raise ValueError("At least five complete observations are required.")
    X, Y = d[xs].to_numpy(float), d[ys].to_numpy(float)
    A, B = _double_center_distance(X), _double_center_distance(Y)
    def dcorr_from_b(Bmat: np.ndarray) -> float:
        dc2 = float(np.mean(A * Bmat))
        vx2 = float(np.mean(A * A)); vy2 = float(np.mean(Bmat * Bmat))
        if vx2 <= 0 or vy2 <= 0:
            return 0.0
        r2 = max(0.0, dc2 / math.sqrt(vx2 * vy2))
        return float(math.sqrt(min(1.0, r2)))
    observed = dcorr_from_b(B)
    rng = np.random.default_rng(seed); perms=max(99,int(permutations))
    null = np.empty(perms)
    for i in range(perms):
        pidx=rng.permutation(len(d)); null[i]=dcorr_from_b(B[np.ix_(pidx,pidx)])
    p = float((1+np.sum(null>=observed))/(perms+1))
    summary = pd.DataFrame([{"distance_correlation": observed, "permutation_p_value": p, "n": len(d), "x_dimension": len(xs), "y_dimension": len(ys), "permutations": perms}])
    return AnalysisResult({"Distance correlation": summary}, pd.DataFrame(), {"x_columns": xs, "y_columns": ys, "seed": seed, "permutations": perms}, [], null)


def energy_two_sample_test(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    group: str,
    permutations: int = 999,
    seed: int = 42,
) -> AnalysisResult:
    """Multivariate two-sample energy-distance permutation test."""
    from scipy.spatial.distance import cdist
    cols=list(columns)
    if not cols:
        raise ValueError("Select at least one numeric variable.")
    d=df[cols+[group]].copy(); d[cols]=d[cols].apply(pd.to_numeric, errors="coerce"); d=d.dropna()
    levels=list(pd.Series(d[group]).drop_duplicates())
    if len(levels)!=2:
        raise ValueError("Energy two-sample test requires exactly two groups.")
    X=d[cols].to_numpy(float); labels=(d[group].to_numpy()==levels[1]).astype(int)
    # standardise to prevent unit scale from arbitrarily dominating the distance.
    sd=np.nanstd(X,axis=0,ddof=1); sd=np.where(sd>0,sd,1.0); X=(X-np.nanmean(X,axis=0))/sd
    D=cdist(X,X)
    def estat(lbl):
        a=np.where(lbl==0)[0]; b=np.where(lbl==1)[0]
        if len(a)==0 or len(b)==0: return np.nan
        return float(2*D[np.ix_(a,b)].mean()-D[np.ix_(a,a)].mean()-D[np.ix_(b,b)].mean())
    obs=estat(labels); rng=np.random.default_rng(seed); perms=max(99,int(permutations)); null=np.empty(perms)
    for i in range(perms): null[i]=estat(rng.permutation(labels))
    p=float((1+np.sum(null>=obs))/(perms+1))
    summary=pd.DataFrame([{"energy_statistic":obs,"permutation_p_value":p,"n":len(d),"variables":len(cols),"group_1":str(levels[0]),"n_1":int(np.sum(labels==0)),"group_2":str(levels[1]),"n_2":int(np.sum(labels==1)),"permutations":perms}])
    return AnalysisResult({"Energy two-sample":summary},pd.DataFrame(),{"columns":cols,"group":group,"seed":seed,"permutations":perms},["Variables are z-standardised before Euclidean energy distance so arbitrary measurement units do not dominate."],null)


def partial_correlation(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    controls: Sequence[str] = (),
    method: str = "pearson",
) -> AnalysisResult:
    """Pearson or rank-based partial correlation via residualisation."""
    cols=list(dict.fromkeys([x,y,*controls])); d=df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d)<5: raise ValueError("At least five complete observations are required.")
    z=d.copy()
    if method.lower().startswith("spear"):
        z=z.rank(method="average")
    C=sm.add_constant(z[list(controls)],has_constant="add") if controls else np.ones((len(z),1))
    rx=np.asarray(sm.OLS(z[x],C).fit().resid,float); ry=np.asarray(sm.OLS(z[y],C).fit().resid,float)
    r=float(np.corrcoef(rx,ry)[0,1]); dfree=len(z)-len(controls)-2
    t=r*math.sqrt(dfree/max(EPS,1-r*r)) if dfree>0 and abs(r)<1 else np.sign(r)*np.inf
    p=float(2*stats.t.sf(abs(t),dfree)) if dfree>0 else np.nan
    # Fisher-z CI, using partial-correlation effective degrees of freedom.
    if len(z)-len(controls)-3>0 and abs(r)<1:
        se=1/math.sqrt(len(z)-len(controls)-3); zz=np.arctanh(r); lo,hi=np.tanh([zz-1.96*se,zz+1.96*se])
    else: lo=hi=np.nan
    summary=pd.DataFrame([{"partial_correlation":r,"t_statistic":t,"df":dfree,"p_value":p,"ci_95_low":lo,"ci_95_high":hi,"n":len(z),"controls":len(controls),"method":method}])
    return AnalysisResult({"Partial correlation":summary},pd.DataFrame(),{"x":x,"y":y,"controls":list(controls),"method":method},[],None)


def meta_regression(
    df: pd.DataFrame,
    *,
    effect: str,
    standard_error: str,
    predictors: Sequence[str],
    categorical: Sequence[str] = (),
) -> AnalysisResult:
    """Two-stage DerSimonian-Laird random-effects meta-regression.

    Tau-squared is estimated from the intercept-only evidence set and then used
    as an additive between-study variance in inverse-variance WLS. This is a
    transparent workhorse implementation, not a substitute for specialised
    REML/HKSJ meta-regression software when those exact inferential conventions
    are mandated by a protocol.
    """
    from statsmodels.stats.meta_analysis import combine_effects
    cols=list(dict.fromkeys([effect,standard_error,*predictors])); d=df[cols].copy()
    d[effect]=pd.to_numeric(d[effect],errors="coerce"); d[standard_error]=pd.to_numeric(d[standard_error],errors="coerce")
    X=_design_matrix(d,predictors,categorical,add_constant=True)
    joined=pd.concat([d[[effect,standard_error]],X],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(joined)<max(5,X.shape[1]+2): raise ValueError("Too few complete studies for the selected meta-regression.")
    e=joined[effect].to_numpy(float); se=joined[standard_error].to_numpy(float)
    if np.any(se<=0): raise ValueError("Standard errors must be positive.")
    comb=combine_effects(e,se**2,method_re="chi2")
    tau2=max(0.0,float(getattr(comb,"tau2",0.0)))
    XX=joined.drop(columns=[effect,standard_error]).astype(float); w=1/(se**2+tau2)
    fit=sm.WLS(e,XX,weights=w).fit()
    coef=_coef_table(fit)
    summary=pd.DataFrame([{"studies":len(joined),"tau_squared_DL":tau2,"aic":float(fit.aic),"bic":float(fit.bic),"r_squared_weighted":float(fit.rsquared)}])
    return AnalysisResult({"Meta-regression coefficients":coef,"Meta-regression fit":summary},pd.DataFrame(),{"effect":effect,"standard_error":standard_error,"predictors":list(predictors),"categorical":list(categorical)},["Tau² uses a DerSimonian-Laird/intercept-only method-of-moments estimate; use specialised REML/HKSJ meta-regression when a protocol requires those exact conventions."],fit)


def parsed_numeric_audit(
    df: pd.DataFrame,
    *,
    raw_column: str,
    normalised_column: str,
    flag_column: str | None = None,
    direct_flags: Sequence[str] = ("numeric_factor",),
) -> AnalysisResult:
    """Audit raw-to-normalised numeric parsing and analytical-sample accounting.

    This is deliberately generic. It catches situations where a manuscript says
    N=182 while QC counts 184 usable normalised values because two values were
    converted from a percentage/increment representation rather than entered as
    direct factors.
    """
    required=[raw_column,normalised_column]+([flag_column] if flag_column else [])
    missing=[c for c in required if c not in df]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    d=df[required].copy(); norm=pd.to_numeric(d[normalised_column],errors="coerce")
    raw_nonmissing=d[raw_column].notna() & d[raw_column].astype(str).str.strip().ne("")
    norm_nonmissing=norm.notna()
    if flag_column:
        flags=d[flag_column].astype("string").fillna("<missing>")
        direct=norm_nonmissing & flags.isin(list(direct_flags))
        converted=norm_nonmissing & ~flags.isin(list(direct_flags))
        freq=flags.value_counts(dropna=False).rename_axis("parsing_flag").reset_index(name="rows")
        byflag=pd.DataFrame({"flag":flags,"normalised_nonmissing":norm_nonmissing}).groupby("flag",dropna=False).agg(rows=("flag","size"),normalised_values=("normalised_nonmissing","sum")).reset_index()
    else:
        direct=norm_nonmissing; converted=pd.Series(False,index=d.index); freq=byflag=pd.DataFrame()
    summary=pd.DataFrame([{
        "rows":len(d),"raw_nonmissing":int(raw_nonmissing.sum()),"normalised_nonmissing":int(norm_nonmissing.sum()),
        "direct_normalised":int(direct.sum()),"converted_or_non_direct_normalised":int(converted.sum()),
        "raw_without_normalisation":int((raw_nonmissing & ~norm_nonmissing).sum()),
        "normalised_without_raw":int((norm_nonmissing & ~raw_nonmissing).sum()),
    }])
    cases=d.copy(); cases["__normalised_numeric__"]=norm; cases["__direct__"]=direct; cases["__converted_or_non_direct__"]=converted
    cases["__raw_without_normalisation__"]=raw_nonmissing & ~norm_nonmissing
    cases=cases.loc[cases[["__converted_or_non_direct__","__raw_without_normalisation__"]].any(axis=1)].copy()
    tables={"Sample accounting":summary,"Exceptional cases":cases.reset_index().rename(columns={"index":"row_index"})}
    if not byflag.empty: tables["Parsing flags"]=byflag
    return AnalysisResult(tables,pd.DataFrame(),{"raw_column":raw_column,"normalised_column":normalised_column,"flag_column":flag_column,"direct_flags":list(direct_flags)},[],None)


METHOD_CATALOGUE = pd.concat([
    METHOD_CATALOGUE,
    pd.DataFrame([
        ["Brunner-Munzel", "two independent groups", "stochastic equality without equal-variance/shape assumption", "confirmatory", "rare"],
        ["Jonckheere-Terpstra", "ordered groups", "monotone ordered alternative with permutation inference", "confirmatory", "rare"],
        ["Quade rank test", "complete blocks / repeated measures", "ranked block comparison using block-range information", "confirmatory", "very rare"],
        ["Cochran Q", "matched binary repeated measures", "omnibus difference in paired binary rates", "confirmatory", "rare"],
        ["McNemar exact", "paired binary", "within-pair marginal change", "confirmatory", "rare"],
        ["Bowker symmetry", "paired multicategory", "multicategory generalisation of McNemar", "confirmatory", "very rare"],
        ["Distance correlation", "numeric / multivariate", "arbitrary nonlinear dependence with permutation test", "confirmatory", "very rare"],
        ["Energy two-sample", "multivariate two-group", "distributional equality beyond means/covariances", "confirmatory", "very rare"],
        ["Partial correlation", "numeric", "linear/rank association conditional on controls", "confirmatory", "specialised"],
        ["Random-effects meta-regression", "study-level evidence", "moderators of heterogeneous effects", "confirmatory", "specialised"],
        ["Raw-normalised parsing audit", "curated/parsed numeric data", "reconcile direct, converted and unparseable sample counts", "quality control", "specialised"],
        ["Dirichlet component-wise log-alpha", "positive composition", "component-specific covariate effects with LR block tests", "confirmatory", "specialised"],
        ["Benzécri-corrected MCA + Ward", "categorical multivariate", "corrected inertia plus low-dimensional Ward segmentation", "confirmatory/exploratory", "specialised"],
        ["Latent-class model selection", "categorical indicators", "automatic K sweep with AIC/BIC selection", "confirmatory/robustness", "specialised"],
        ["Plackett-Luce model selection", "rankings", "single versus finite-mixture ranking models with AIC/BIC", "confirmatory/robustness", "specialised"],
    ], columns=METHOD_CATALOGUE.columns),
], ignore_index=True)
