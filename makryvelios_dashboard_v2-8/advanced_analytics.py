"""Advanced postdoctoral-grade analytics with auditable tabular outputs.

The functions in this module are UI-independent and deterministic when a seed
is supplied. They intentionally return tables rather than opaque estimator
objects so every result can be displayed, exported and tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ClusteringOutput:
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    diagnostics: pd.DataFrame
    embedding: pd.DataFrame
    interpretation: list[str]
    selected_k: int | None


def _scaled_matrix(df: pd.DataFrame, columns: Sequence[str], scaling: str):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    cols = [c for c in dict.fromkeys(columns) if c in df]
    if not cols:
        raise ValueError("Select at least one clustering variable.")
    raw = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = [c for c in cols if raw[c].notna().sum() >= 3 and raw[c].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError("The selected variables are empty or constant after numeric conversion.")
    raw = raw[usable]
    matrix = SimpleImputer(strategy="median").fit_transform(raw)
    if scaling == "Robust (median/IQR)":
        scaler = RobustScaler()
    elif scaling == "Min-max [0,1]":
        scaler = MinMaxScaler()
    elif scaling == "None":
        scaler = None
    else:
        scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix) if scaler else matrix
    return raw, np.asarray(scaled, float), usable


def _cluster_labels(matrix: np.ndarray, method: str, k: int, linkage: str, eps: float, min_samples: int, seed: int):
    if method == "Hierarchical agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(matrix), None
    if method == "Gaussian mixture":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed).fit(matrix)
        return model.predict(matrix), model
    if method == "DBSCAN":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix), None
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, n_init=30, random_state=seed).fit(matrix)
    return model.labels_, model


def _cluster_quality(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    mask = labels >= 0
    unique = np.unique(labels[mask])
    if mask.sum() < 3 or len(unique) < 2 or len(unique) >= mask.sum():
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(matrix[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(matrix[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(matrix[mask], labels[mask])),
    }


def advanced_clustering(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "K-means",
    clusters: int = 4,
    scaling: str = "Standard (z-score)",
    automatic_k: bool = True,
    max_k: int = 10,
    linkage: str = "ward",
    eps: float = .7,
    min_samples: int = 5,
    seed: int = 42,
) -> ClusteringOutput:
    """Cluster one or many numeric variables with comparative diagnostics."""
    raw, matrix, usable = _scaled_matrix(df, columns, scaling)
    n = len(raw)
    if n < 5:
        raise ValueError("At least five observations are required for clustering.")
    max_allowed = min(max(2, int(max_k)), 20, n - 1)
    diagnostics: list[dict] = []
    selected_k: int | None = None
    model = None
    if method == "DBSCAN":
        labels, model = _cluster_labels(matrix, method, 0, linkage, eps, min_samples, seed)
        quality = _cluster_quality(matrix, labels)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        diagnostics.append({"method": method, "clusters": cluster_count, "noise_observations": int(np.sum(labels == -1)), **quality, "aic": np.nan, "bic": np.nan, "selected": True})
    else:
        candidate_ks = range(2, max_allowed + 1) if automatic_k else [min(max(2, int(clusters)), max_allowed)]
        candidates: dict[int, tuple[np.ndarray, object | None]] = {}
        for k in candidate_ks:
            lab, candidate_model = _cluster_labels(matrix, method, k, linkage, eps, min_samples, seed)
            candidates[k] = (lab, candidate_model)
            quality = _cluster_quality(matrix, lab)
            diagnostics.append({
                "method": method, "clusters": k, "noise_observations": 0, **quality,
                "aic": float(candidate_model.aic(matrix)) if method == "Gaussian mixture" else np.nan,
                "bic": float(candidate_model.bic(matrix)) if method == "Gaussian mixture" else np.nan,
                "selected": False,
            })
        diagnostic_frame = pd.DataFrame(diagnostics)
        if automatic_k and diagnostic_frame.silhouette.notna().any():
            selected_k = int(diagnostic_frame.loc[diagnostic_frame.silhouette.idxmax(), "clusters"])
        elif method == "Gaussian mixture" and diagnostic_frame.bic.notna().any():
            selected_k = int(diagnostic_frame.loc[diagnostic_frame.bic.idxmin(), "clusters"])
        else:
            selected_k = int(list(candidate_ks)[0])
        labels, model = candidates[selected_k]
        for row in diagnostics:
            row["selected"] = row["clusters"] == selected_k
    diagnostics_df = pd.DataFrame(diagnostics)

    # Perturbation stability on a bounded common subsample. This tests whether
    # small measurement noise materially changes the partition.
    from sklearn.metrics import adjusted_rand_score
    stability_rng = np.random.default_rng(seed + 10_003)
    subset = stability_rng.choice(n, size=min(n, 1_000), replace=False)
    stability_matrix = matrix[subset]
    stability_k = int(diagnostics_df.loc[diagnostics_df.selected, "clusters"].iloc[0]) if method != "DBSCAN" else 0
    reference_labels, _ = _cluster_labels(stability_matrix, method, stability_k, linkage, eps, min_samples, seed)
    stability_scores = []
    for repeat in range(10):
        perturbed = stability_matrix + stability_rng.normal(0, .02, size=stability_matrix.shape)
        perturbed_labels, _ = _cluster_labels(perturbed, method, stability_k, linkage, eps, min_samples, seed + repeat + 1)
        stability_scores.append(adjusted_rand_score(reference_labels, perturbed_labels))
    diagnostics_df["perturbation_stability_ari"] = np.nan
    diagnostics_df.loc[diagnostics_df.selected, "perturbation_stability_ari"] = float(np.mean(stability_scores))

    cluster_names = np.where(labels == -1, "Noise / outlier", np.char.add("Cluster ", labels.astype(str)))
    assignments = pd.DataFrame({"row_index": df.index, "cluster": labels, "cluster_label": cluster_names})
    original = raw.copy()
    original["cluster"] = labels
    original["cluster_label"] = cluster_names
    means = original.groupby(["cluster", "cluster_label"], dropna=False)[usable].mean().reset_index()
    medians = original.groupby(["cluster", "cluster_label"], dropna=False)[usable].median().reset_index()
    counts = original.groupby(["cluster", "cluster_label"], dropna=False).size().reset_index(name="observations")
    profiles = counts.merge(means, on=["cluster", "cluster_label"], how="left", suffixes=("", "_mean"))
    for c in usable:
        profiles = profiles.rename(columns={c: f"mean_{c}"})
        profiles[f"median_{c}"] = medians.set_index(["cluster", "cluster_label"]).loc[
            list(zip(profiles.cluster, profiles.cluster_label)), c
        ].to_numpy()
    profiles["share_of_sample"] = profiles.observations / len(original)

    from sklearn.decomposition import PCA
    components = min(2, matrix.shape[1], matrix.shape[0])
    scores = PCA(n_components=components, random_state=seed).fit_transform(matrix)
    embedding = pd.DataFrame({"row_index": df.index, "dimension_1": scores[:, 0], "cluster": labels, "cluster_label": cluster_names})
    embedding["dimension_2"] = scores[:, 1] if components > 1 else 0.0
    if len(usable) == 1:
        embedding["dimension_1"] = matrix[:, 0]
        embedding["dimension_2"] = np.linspace(-.025, .025, len(matrix))
        embedding["projection_note"] = f"One-dimensional projection of {usable[0]}; vertical jitter is visual only."
    else:
        embedding["projection_note"] = "Two-dimensional PCA projection of the scaled clustering variables."

    quality_row = diagnostics_df.loc[diagnostics_df.selected].iloc[0]
    n_clusters = int(quality_row.clusters)
    comments = [
        f"{method} produced {n_clusters} substantive cluster(s) from {len(usable)} variable(s) and {n:,} observations.",
        f"Scaling rule: {scaling}. Scaling changes the geometry of distance-based clustering and must be reported.",
    ]
    if automatic_k and method != "DBSCAN":
        comments.append(f"The selected solution uses k={selected_k}, maximising the silhouette coefficient among the evaluated candidate values.")
    if np.isfinite(quality_row.silhouette):
        value = float(quality_row.silhouette)
        strength = "strong" if value >= .50 else "moderate" if value >= .25 else "weak/overlapping"
        comments.append(f"Silhouette={value:.3f}, indicating {strength} separation under this representation.")
    stability = float(diagnostics_df.loc[diagnostics_df.selected, "perturbation_stability_ari"].iloc[0])
    stability_label = "high" if stability >= .90 else "moderate" if stability >= .70 else "low"
    comments.append(f"Perturbation stability ARI={stability:.3f} ({stability_label}); this measures partition agreement after ten small-noise refits on a bounded common subsample.")
    if method == "DBSCAN":
        comments.append(f"DBSCAN labelled {int(quality_row.noise_observations):,} observations as noise. Results are sensitive to eps and min_samples.")
    comments.append("Cluster labels are descriptive typologies, not naturally occurring causal classes. Validate stability, substantive coherence and sensitivity to scaling/variables before publication.")
    return ClusteringOutput(assignments, profiles, diagnostics_df, embedding, comments, selected_k)


def predictive_model_comparison(
    df: pd.DataFrame,
    y: str,
    x_vars: Sequence[str],
    folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Compare interpretable and nonlinear regressors using honest CV metrics."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xs = [c for c in dict.fromkeys(x_vars) if c in df and c != y][:200]
    if not xs:
        raise ValueError("Select at least one predictor.")
    X = df[xs].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Y = pd.to_numeric(df[y], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = Y.notna()
    X, Y = X.loc[keep], Y.loc[keep]
    if len(Y) < 40:
        raise ValueError("At least 40 observations are required for model comparison.")
    folds = min(max(3, int(folds)), 10, len(Y) // 10)
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    linear = lambda model: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    tree = lambda model: make_pipeline(SimpleImputer(strategy="median"), model)
    models = {
        "OLS": linear(LinearRegression()),
        "Ridge": linear(Ridge(alpha=1.0)),
        "Lasso": linear(Lasso(alpha=.01, max_iter=20_000, random_state=seed)),
        "Elastic Net": linear(ElasticNet(alpha=.01, l1_ratio=.5, max_iter=20_000, random_state=seed)),
        "Random forest": tree(RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
        "Extra trees": tree(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "Gradient boosting": tree(GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.04, random_state=seed)),
    }
    rows, predictions = [], pd.DataFrame({"row_index": Y.index, "observed": Y.to_numpy()})
    prediction_cache: dict[str, np.ndarray] = {}
    for name, model in models.items():
        pred = cross_val_predict(model, X, Y, cv=cv, n_jobs=None)
        prediction_cache[name] = pred
        predictions[name] = pred
        rows.append({
            "model": name, "folds": folds, "n": len(Y), "predictors": len(xs),
            "cross_validated_r_squared": r2_score(Y, pred),
            "cross_validated_rmse": np.sqrt(mean_squared_error(Y, pred)),
            "cross_validated_mae": mean_absolute_error(Y, pred),
        })
    performance = pd.DataFrame(rows).sort_values(["cross_validated_rmse", "cross_validated_mae"])
    best_name = str(performance.iloc[0].model)
    best = models[best_name].fit(X, Y)
    importance = permutation_importance(best, X, Y, n_repeats=7, random_state=seed, scoring="neg_root_mean_squared_error", n_jobs=None)
    importances = pd.DataFrame({
        "variable": xs, "permutation_importance_mean": importance.importances_mean,
        "permutation_importance_sd": importance.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    comments = [
        f"{best_name} has the lowest cross-validated RMSE ({performance.iloc[0].cross_validated_rmse:.4g}) among the seven candidate models.",
        f"Performance is based on {folds}-fold shuffled cross-validation across {len(Y):,} observations; it is not in-sample fit.",
        "Permutation importance measures predictive contribution conditional on this fitted model and predictor set; it is not a causal effect or a structural coefficient.",
        "If observations are grouped by region, organisation or time, ordinary random folds may be optimistic. Use a grouped or time-ordered validation design for final publication.",
    ]
    return performance, predictions, importances, comments


def panel_model_suite(
    df: pd.DataFrame,
    entity: str,
    time: str,
    y: str,
    x_vars: Sequence[str],
    aggregation: str = "Mean",
    covariance: str = "Clustered by entity",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Pooled, two-way fixed-effects and random-effects panel comparison."""
    try:
        from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("The linearmodels package is required for panel estimation.") from exc
    import statsmodels.api as sm

    xs = [c for c in dict.fromkeys(x_vars) if c in df and c not in {entity, time, y}]
    if not xs:
        raise ValueError("Select at least one panel predictor.")
    d = df[[entity, time, y, *xs]].copy()
    d = d.dropna(subset=[entity, time])
    for c in [y, *xs]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    agg_fn = {"Sum": "sum", "Median": "median"}.get(aggregation, "mean")
    counts = d.groupby([entity, time]).size().rename("source_rows")
    panel = d.groupby([entity, time])[[y, *xs]].agg(agg_fn).join(counts).dropna().reset_index()
    if panel[entity].nunique() < 2 or panel[time].nunique() < 2:
        raise ValueError("Panel models require at least two entities and two time periods.")
    panel = panel.set_index([entity, time]).sort_index()
    Y = panel[y].astype(float)
    X = panel[xs].astype(float)
    X_const = sm.add_constant(X, has_constant="add")
    cov_kw = {"cov_type": "clustered", "cluster_entity": True} if covariance == "Clustered by entity" else {"cov_type": "robust" if covariance == "Robust" else "unadjusted"}
    fitted = {}
    errors = []
    estimators = {
        "Pooled OLS": lambda: PooledOLS(Y, X_const).fit(**cov_kw),
        "Two-way fixed effects": lambda: PanelOLS(Y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(**cov_kw),
        "Random effects": lambda: RandomEffects(Y, X_const).fit(**cov_kw),
    }
    coef_rows, fit_rows = [], []
    for name, factory in estimators.items():
        try:
            result = factory(); fitted[name] = result
            ci = result.conf_int()
            for term in result.params.index:
                coef_rows.append({
                    "model": name, "term": term, "coefficient": result.params[term],
                    "std_error": result.std_errors[term], "statistic": result.tstats[term],
                    "p_value": result.pvalues[term], "ci_95_low": ci.loc[term, "lower"],
                    "ci_95_high": ci.loc[term, "upper"],
                })
            fit_rows.append({
                "model": name, "n": int(result.nobs), "entities": panel.index.get_level_values(0).nunique(),
                "periods": panel.index.get_level_values(1).nunique(), "r_squared": result.rsquared,
                "r_squared_within": getattr(result, "rsquared_within", np.nan),
                "r_squared_between": getattr(result, "rsquared_between", np.nan),
                "r_squared_overall": getattr(result, "rsquared_overall", np.nan),
                "log_likelihood": result.loglik,
            })
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not fitted:
        raise ValueError("All panel specifications failed: " + "; ".join(errors))
    coefficients = pd.DataFrame(coef_rows)
    fit = pd.DataFrame(fit_rows)
    hausman = pd.DataFrame()
    if "Two-way fixed effects" in fitted and "Random effects" in fitted:
        fe, re = fitted["Two-way fixed effects"], fitted["Random effects"]
        common = [c for c in fe.params.index if c in re.params.index and c != "const"]
        if common:
            diff = fe.params[common] - re.params[common]
            cov_diff = fe.cov.loc[common, common] - re.cov.loc[common, common]
            statistic = float(diff.to_numpy() @ np.linalg.pinv(cov_diff.to_numpy()) @ diff.to_numpy())
            statistic = max(statistic, 0.0)
            p_value = float(stats.chi2.sf(statistic, len(common)))
            hausman = pd.DataFrame([{"test": "Hausman FE versus RE", "statistic": statistic, "degrees_freedom": len(common), "p_value": p_value, "common_coefficients": "; ".join(common)}])
    prepared = panel.reset_index()
    comments = [
        f"The source was aggregated to {len(prepared):,} unique {entity}–{time} cells using {aggregation.lower()} values; source_rows records the number of original rows in each cell.",
        "Two-way fixed effects absorb time-invariant entity heterogeneity and common time shocks. Their coefficients are identified from within-entity change.",
        "Random effects require the unobserved entity effect to be uncorrelated with every included regressor; this assumption is stronger than fixed effects.",
    ]
    if not hausman.empty:
        p = float(hausman.p_value.iloc[0])
        comments.append(f"Hausman p={p:.4g}. " + ("The coefficient difference rejects the random-effects orthogonality restriction at 5%, favouring fixed effects." if p < .05 else "The test does not reject random effects at 5%, although substantive assumptions still control model choice."))
    if errors:
        comments.append("Some estimators were unavailable for this design: " + "; ".join(errors))
    comments.append("Panel estimates remain associational unless treatment timing or another defensible identification strategy establishes a causal estimand.")
    return fit, coefficients, hausman, prepared, comments
