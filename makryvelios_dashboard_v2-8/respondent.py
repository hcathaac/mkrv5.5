"""Respondent-level preference analytics for empirical ITA weight distributions."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


@dataclass
class RespondentOutput:
    respondents: pd.DataFrame
    raw_values: pd.DataFrame
    normalised_weights: pd.DataFrame
    criterion_summary: pd.DataFrame
    correlation: pd.DataFrame
    pca_loadings: pd.DataFrame
    pca_variance: pd.DataFrame
    cluster_profiles: pd.DataFrame
    subgroup_tests: pd.DataFrame
    diagnostics: pd.DataFrame
    kendall_w: float
    settings: dict


def _kendall_w(values: np.ndarray) -> float:
    """Kendall's coefficient of concordance with tie correction."""
    m, n = values.shape
    if m < 2 or n < 2:
        return np.nan
    ranked = np.asarray([stats.rankdata(-row, method="average") for row in values])
    rank_sums = ranked.sum(axis=0)
    s = float(np.square(rank_sums - rank_sums.mean()).sum())
    tie_term = 0.0
    for row in values:
        _, counts = np.unique(row, return_counts=True)
        tie_term += float(np.sum(counts**3 - counts))
    denominator = m * m * (n**3 - n) - m * tie_term
    return float(12 * s / denominator) if denominator > 0 else np.nan


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def _subgroup_tests(weights: pd.DataFrame, groups: pd.Series, criteria: Sequence[str]) -> pd.DataFrame:
    frame = weights.copy()
    frame["group"] = groups.astype("string").fillna("Missing").to_numpy()
    rows: list[dict] = []
    for criterion in criteria:
        samples = [g[criterion].dropna().to_numpy(float) for _, g in frame.groupby("group")]
        samples = [sample for sample in samples if len(sample) >= 2]
        if len(samples) < 2:
            continue
        statistic, p_value = stats.kruskal(*samples)
        n = sum(map(len, samples)); k = len(samples)
        epsilon = max(0.0, float((statistic - k + 1) / (n - k))) if n > k else np.nan
        rows.append({
            "criterion": criterion, "test": "Kruskal-Wallis", "groups": k, "respondents": n,
            "statistic": float(statistic), "p_value": float(p_value), "epsilon_squared": epsilon,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted_bh"] = multipletests(out.p_value, method="fdr_bh")[1]
        out["significant_5pct_bh"] = out.p_adjusted_bh < .05
    return out


def analyse_respondents(
    data: pd.DataFrame,
    *,
    respondent_id: str,
    weight_columns: Sequence[str],
    group_column: str | None = None,
    missing: str = "complete",
    seed: int = 42,
) -> RespondentOutput:
    """Analyse empirical respondent preferences without changing the source data."""
    if respondent_id not in data or len(weight_columns) < 2:
        raise ValueError("Select a respondent identifier and at least two criterion-weight variables.")
    missing_columns = [column for column in weight_columns if column not in data]
    if missing_columns:
        raise ValueError("Missing selected variables: " + ", ".join(missing_columns))
    raw = data[list(weight_columns)].apply(pd.to_numeric, errors="coerce")
    source_rows = len(raw)
    duplicate_ids = int(data[respondent_id].astype(str).duplicated().sum())
    negative_rows = raw.lt(0).any(axis=1)
    if missing == "median":
        usable = raw.loc[~negative_rows].copy()
        usable = usable.fillna(usable.median())
        valid = usable.notna().all(axis=1) & usable.sum(axis=1).gt(0)
    else:
        usable = raw.copy()
        valid = ~negative_rows & usable.notna().all(axis=1) & usable.sum(axis=1).gt(0)
    usable = usable.loc[valid].copy()
    if len(usable) < 4:
        raise ValueError("At least four valid respondent records are required after missing/invalid-row handling.")
    criteria = [f"C{i + 1}" for i in range(len(weight_columns))]
    raw_values = usable.copy(); raw_values.columns = criteria
    normalised = raw_values.div(raw_values.sum(axis=1), axis=0)
    respondent_ids = data.loc[usable.index, respondent_id].astype(str).reset_index(drop=True)
    raw_values = raw_values.reset_index(drop=True)
    normalised = normalised.reset_index(drop=True)
    source_positions = data.index.get_indexer(usable.index) + 1
    respondent_table = pd.DataFrame({"respondent_id": respondent_ids, "source_row": source_positions})
    if group_column and group_column in data:
        respondent_table["group"] = data.loc[usable.index, group_column].astype("string").fillna("Missing").reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    summary_rows: list[dict] = []
    for criterion, source in zip(criteria, weight_columns):
        values = normalised[criterion].to_numpy(float)
        low, high = _bootstrap_mean_ci(values, rng)
        summary_rows.append({
            "criterion": criterion, "source_column": source, "respondents": len(values),
            "mean_weight": float(values.mean()), "median_weight": float(np.median(values)),
            "std_dev": float(values.std(ddof=1)), "iqr": float(np.quantile(values, .75) - np.quantile(values, .25)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
        })
    criterion_summary = pd.DataFrame(summary_rows)
    correlation = normalised.corr(method="spearman")
    kendall_w = _kendall_w(raw_values.to_numpy(float))

    scaled = StandardScaler().fit_transform(normalised)
    max_k = min(6, len(normalised) - 1)
    best_k, best_silhouette, labels = 1, np.nan, np.zeros(len(normalised), dtype=int)
    for k in range(2, max_k + 1):
        candidate = KMeans(n_clusters=k, random_state=int(seed), n_init=20).fit_predict(scaled)
        if len(np.unique(candidate)) < 2:
            continue
        score = silhouette_score(scaled, candidate)
        if best_k == 1 or score > best_silhouette:
            best_k, best_silhouette, labels = k, float(score), candidate
    respondent_table["preference_cluster"] = labels + 1
    cluster_frame = normalised.copy(); cluster_frame["preference_cluster"] = labels + 1
    cluster_profiles = cluster_frame.groupby("preference_cluster", as_index=False).agg(
        {**{criterion: "mean" for criterion in criteria}}
    )
    counts = respondent_table.preference_cluster.value_counts().rename("respondents")
    cluster_profiles.insert(1, "respondents", cluster_profiles.preference_cluster.map(counts))

    components = min(2, len(criteria), len(normalised))
    pca = PCA(n_components=components, random_state=int(seed)).fit(scaled)
    scores = pca.transform(scaled)
    respondent_table["PC1"] = scores[:, 0]
    respondent_table["PC2"] = scores[:, 1] if components > 1 else 0.0
    pca_loadings = pd.DataFrame(pca.components_.T, index=criteria, columns=[f"PC{i + 1}" for i in range(components)]).reset_index(names="criterion")
    pca_variance = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    subgroup = _subgroup_tests(normalised, respondent_table["group"], criteria) if "group" in respondent_table else pd.DataFrame()
    diagnostics = pd.DataFrame([
        {"check": "Source rows", "value": source_rows},
        {"check": "Valid respondents", "value": len(normalised)},
        {"check": "Excluded rows", "value": source_rows - len(normalised)},
        {"check": "Duplicate respondent identifiers", "value": duplicate_ids},
        {"check": "Rows with negative selected values", "value": int(negative_rows.sum())},
        {"check": "Automatically selected clusters", "value": best_k},
        {"check": "Cluster silhouette", "value": best_silhouette},
        {"check": "Kendall concordance W", "value": kendall_w},
    ])
    return RespondentOutput(
        respondents=respondent_table, raw_values=raw_values, normalised_weights=normalised,
        criterion_summary=criterion_summary, correlation=correlation.reset_index(names="criterion"),
        pca_loadings=pca_loadings, pca_variance=pca_variance, cluster_profiles=cluster_profiles,
        subgroup_tests=subgroup, diagnostics=diagnostics, kendall_w=kendall_w,
        settings={
            "respondent_id": respondent_id, "source_columns": list(weight_columns),
            "criterion_labels": criteria, "group_column": group_column, "missing": missing,
            "seed": int(seed), "valid_respondents": len(normalised),
        },
    )


def respondent_export_bundle(output: RespondentOutput) -> bytes:
    """Create a complete respondent-level analysis and ITA bridge package."""
    files = {
        "respondents.csv": output.respondents,
        "raw_values.csv": output.raw_values,
        "normalised_empirical_weights.csv": output.normalised_weights,
        "criterion_summary.csv": output.criterion_summary,
        "spearman_correlation.csv": output.correlation,
        "pca_loadings.csv": output.pca_loadings,
        "pca_variance.csv": output.pca_variance,
        "cluster_profiles.csv": output.cluster_profiles,
        "subgroup_tests.csv": output.subgroup_tests,
        "diagnostics.csv": output.diagnostics,
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in files.items():
            archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("ita_empirical_weight_profile.json", json.dumps(output.settings, indent=2, ensure_ascii=False))
        archive.writestr("README.txt", (
            "The normalised empirical weight matrix contains one row per valid respondent and one column per mapped criterion. "
            "It can be sampled by Hybrid ITA-RW. Raw responses are retained separately. Normalisation creates relative weights; "
            "it does not establish that Likert items form a psychometric scale or that observed differences are causal."
        ))
    return payload.getvalue()
