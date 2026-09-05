"""Frontier analytical methods added in v5.8.0.

All functions are additive and designed to fail safely. Numerical results do not
require an external LLM. Optional third-party accelerators (DuckDB/Arrow/SHAP)
are used when installed and otherwise expose clear fallbacks.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import invgamma
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ParetoResult:
    frontier: pd.DataFrame
    project_frequency: pd.DataFrame
    selections: dict[str, np.ndarray]
    objective_columns: list[str]
    effective_cost: pd.Series


@dataclass
class CausalResult:
    estimate: pd.DataFrame
    balance: pd.DataFrame
    unit_diagnostics: pd.DataFrame
    assumptions: list[str]


@dataclass
class BayesianResult:
    summary: pd.DataFrame
    draws: pd.DataFrame
    predictive: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass
class ExplainableResult:
    performance: pd.DataFrame
    global_importance: pd.DataFrame
    local_explanation: pd.DataFrame
    predictions: pd.DataFrame
    backend: str


def _simplex_weights(k: int, resolution: int = 8, max_points: int = 120) -> list[np.ndarray]:
    """Deterministic simplex grid for 2-4 objectives."""
    if k < 2:
        return [np.ones(k)]
    resolution = max(2, int(resolution))
    weights: list[np.ndarray] = []
    if k == 2:
        for i in range(resolution + 1):
            weights.append(np.array([i, resolution - i], dtype=float) / resolution)
    else:
        def rec(prefix, remaining, slots):
            if len(weights) >= max_points:
                return
            if slots == 1:
                arr = np.array(prefix + [remaining], dtype=float) / resolution
                weights.append(arr)
                return
            for value in range(remaining + 1):
                rec(prefix + [value], remaining - value, slots - 1)
        rec([], resolution, k)
    # include equal weights and corners, then deduplicate
    weights.extend([np.ones(k) / k] + [np.eye(k)[i] for i in range(k)])
    unique = []
    seen = set()
    for w in weights:
        key = tuple(np.round(w / (w.sum() or 1), 8))
        if key not in seen:
            seen.add(key)
            unique.append(np.asarray(key, dtype=float))
    return unique[:max_points]


def pareto_portfolio(
    df: pd.DataFrame,
    *,
    project_id: str,
    objectives: Sequence[str],
    cost_column: str,
    budget: float,
    uncertainty_columns: Mapping[str, str | None] | None = None,
    robust_lambda: float = 0.0,
    cost_uncertainty_column: str | None = None,
    cost_risk_multiplier: float = 0.0,
    resolution: int = 8,
    max_solutions: int = 120,
) -> ParetoResult:
    objectives = list(objectives)
    if len(objectives) < 2:
        raise ValueError("Select at least two objectives for Pareto optimisation.")
    if len(objectives) > 4:
        raise ValueError("The interactive Pareto laboratory supports up to four simultaneous objectives.")
    required = [project_id, cost_column, *objectives]
    if cost_uncertainty_column:
        required.append(cost_uncertainty_column)
    for col in (uncertainty_columns or {}).values():
        if col:
            required.append(col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(set(missing))))

    work = df[required].copy()
    for c in required:
        if c != project_id:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[project_id, cost_column, *objectives]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No complete project rows remain after mapping objectives and costs.")

    base_cost = work[cost_column].to_numpy(float)
    if np.any(base_cost < 0):
        raise ValueError("Project costs must be non-negative.")
    effective_cost = base_cost.copy()
    if cost_uncertainty_column:
        unc = np.nan_to_num(work[cost_uncertainty_column].to_numpy(float), nan=0.0)
        effective_cost = effective_cost + max(0.0, float(cost_risk_multiplier)) * np.maximum(0.0, unc)

    raw = work[objectives].to_numpy(float)
    robust = raw.copy()
    uncertainty_columns = dict(uncertainty_columns or {})
    for j, objective in enumerate(objectives):
        ucol = uncertainty_columns.get(objective)
        if ucol:
            unc = np.nan_to_num(work[ucol].to_numpy(float), nan=0.0)
            robust[:, j] = raw[:, j] - max(0.0, float(robust_lambda)) * np.maximum(0.0, unc)

    # Min-max normalisation for scalarisation only. Reported objective totals stay in original units.
    mins = np.nanmin(robust, axis=0)
    maxs = np.nanmax(robust, axis=0)
    spans = np.where(np.isclose(maxs, mins), 1.0, maxs - mins)
    norm = (robust - mins) / spans

    weights = _simplex_weights(len(objectives), resolution=resolution, max_points=max_solutions)
    constraint = LinearConstraint(effective_cost.reshape(1, -1), -np.inf, float(budget))
    bounds = Bounds(np.zeros(len(work)), np.ones(len(work)))
    rows = []
    selections: dict[str, np.ndarray] = {}
    seen = set()
    for idx, w in enumerate(weights, start=1):
        utility = norm @ w
        result = milp(c=-utility, integrality=np.ones(len(work)), bounds=bounds, constraints=constraint, options={"disp": False})
        if result.x is None:
            continue
        selected = (np.asarray(result.x) >= 0.5).astype(int)
        key = tuple(np.flatnonzero(selected).tolist())
        if key in seen:
            continue
        seen.add(key)
        solution_id = f"P{len(rows) + 1:03d}"
        selections[solution_id] = selected
        row = {
            "solution_id": solution_id,
            "selected_projects": int(selected.sum()),
            "effective_cost": float(effective_cost @ selected),
            "budget_remaining": float(budget - effective_cost @ selected),
            "scalarised_utility": float(utility @ selected),
        }
        for j, objective in enumerate(objectives):
            row[f"weight_{objective}"] = float(w[j])
            row[f"total_{objective}"] = float(raw[:, j] @ selected)
            row[f"robust_total_{objective}"] = float(robust[:, j] @ selected)
        rows.append(row)

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        raise RuntimeError("No feasible Pareto portfolio was found under the configured budget.")

    # Explicit non-dominance filter in original objective totals.
    obj_cols = [f"total_{o}" for o in objectives]
    vals = frontier[obj_cols].to_numpy(float)
    keep = np.ones(len(frontier), dtype=bool)
    for i in range(len(frontier)):
        if not keep[i]:
            continue
        dominated = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
        if dominated.any():
            keep[i] = False
    frontier = frontier.loc[keep].reset_index(drop=True)
    selections = {sid: selections[sid] for sid in frontier.solution_id}

    freq = np.sum(np.column_stack(list(selections.values())), axis=1) / max(1, len(selections))
    project_frequency = pd.DataFrame({
        "project_id": work[project_id].astype(str),
        "pareto_selection_frequency": freq,
        "effective_cost": effective_cost,
    })
    for objective in objectives:
        project_frequency[objective] = work[objective].to_numpy()
    return ParetoResult(frontier, project_frequency, selections, objectives, pd.Series(effective_cost, index=work.index))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    denom = float(np.sum(w))
    return float(np.sum(x * w) / denom) if denom else np.nan


def _smd(x: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        m1, m0 = np.mean(x[t == 1]), np.mean(x[t == 0])
        v1, v0 = np.var(x[t == 1], ddof=1), np.var(x[t == 0], ddof=1)
    else:
        w1, w0 = weights[t == 1], weights[t == 0]
        x1, x0 = x[t == 1], x[t == 0]
        m1, m0 = _weighted_mean(x1, w1), _weighted_mean(x0, w0)
        v1 = _weighted_mean((x1 - m1) ** 2, w1)
        v0 = _weighted_mean((x0 - m0) ** 2, w0)
    denom = math.sqrt(max(1e-12, (v1 + v0) / 2))
    return float((m1 - m0) / denom)


def causal_aipw(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    folds: int = 5,
    clip: float = 0.01,
) -> CausalResult:
    """Cross-fitted augmented inverse-probability weighted ATE for binary treatment.

    This estimates an ATE under the user's identification assumptions. It does not
    prove exchangeability/no-unmeasured-confounding.
    """
    covariates = list(covariates)
    if not covariates:
        raise ValueError("Select at least one pre-treatment covariate.")
    required = [outcome, treatment, *covariates]
    work = df[required].copy().dropna()
    if len(work) < 40:
        raise ValueError("At least 40 complete observations are required for the cross-fitted causal estimator.")
    y = pd.to_numeric(work[outcome], errors="coerce").to_numpy(float)
    X = work[covariates].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(y) & np.isfinite(X.to_numpy(float)).all(axis=1)
    work = work.loc[valid].reset_index(drop=True)
    y = y[valid]
    Xv = X.loc[valid].to_numpy(float)
    t_raw = work[treatment]
    unique = list(pd.unique(t_raw))
    if len(unique) != 2:
        raise ValueError("Treatment must contain exactly two observed levels.")
    # deterministic mapping: numeric higher value -> treated, otherwise second sorted label -> treated
    if pd.api.types.is_numeric_dtype(t_raw):
        vals = sorted(pd.to_numeric(t_raw).unique())
        t = (pd.to_numeric(t_raw).to_numpy() == vals[-1]).astype(int)
        mapping = f"{vals[0]}→0, {vals[-1]}→1"
    else:
        vals = sorted(map(str, unique))
        t = (t_raw.astype(str).to_numpy() == vals[-1]).astype(int)
        mapping = f"{vals[0]}→0, {vals[-1]}→1"
    if min(t.sum(), (1 - t).sum()) < 15:
        raise ValueError("Each treatment arm needs at least 15 complete observations.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv)
    k = min(max(2, int(folds)), int(min(t.sum(), (1 - t).sum())))
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=580)
    p = np.zeros(len(work), dtype=float)
    m1 = np.zeros(len(work), dtype=float)
    m0 = np.zeros(len(work), dtype=float)
    for train, test in splitter.split(Xs, t):
        propensity = LogisticRegression(max_iter=2000, solver="lbfgs")
        propensity.fit(Xs[train], t[train])
        p[test] = propensity.predict_proba(Xs[test])[:, 1]
        reg1 = Ridge(alpha=1.0).fit(Xs[train][t[train] == 1], y[train][t[train] == 1])
        reg0 = Ridge(alpha=1.0).fit(Xs[train][t[train] == 0], y[train][t[train] == 0])
        m1[test] = reg1.predict(Xs[test])
        m0[test] = reg0.predict(Xs[test])
    p = np.clip(p, float(clip), 1 - float(clip))
    psi = m1 - m0 + t * (y - m1) / p - (1 - t) * (y - m0) / (1 - p)
    ate = float(np.mean(psi))
    se = float(np.std(psi - ate, ddof=1) / math.sqrt(len(psi)))
    z = 1.959963984540054
    estimate = pd.DataFrame([{
        "estimand": "ATE",
        "estimator": "cross-fitted AIPW / doubly robust",
        "estimate": ate,
        "standard_error": se,
        "ci95_low": ate - z * se,
        "ci95_high": ate + z * se,
        "n": len(work),
        "treated_n": int(t.sum()),
        "control_n": int((1 - t).sum()),
        "treatment_mapping": mapping,
        "propensity_min": float(p.min()),
        "propensity_max": float(p.max()),
        "overlap_share_0.05_0.95": float(np.mean((p >= .05) & (p <= .95))),
    }])
    ipw = t / p + (1 - t) / (1 - p)
    bal_rows = []
    for j, cov in enumerate(covariates):
        bal_rows.append({
            "covariate": cov,
            "smd_unweighted": _smd(Xv[:, j], t),
            "smd_ipw": _smd(Xv[:, j], t, ipw),
        })
    balance = pd.DataFrame(bal_rows)
    unit = pd.DataFrame({
        "outcome": y,
        "treatment": t,
        "propensity": p,
        "m1": m1,
        "m0": m0,
        "aipw_score": psi,
        "ipw": ipw,
    })
    assumptions = [
        "Consistency/SUTVA: observed outcomes correspond to the received treatment and interference is absent or negligible.",
        "Conditional exchangeability: all material common causes of treatment and outcome are included among the selected pre-treatment covariates.",
        "Positivity/overlap: every relevant covariate profile has a non-zero probability of both treatment states.",
        "Correct enough nuisance estimation: AIPW is doubly robust but still depends on usable propensity and outcome models.",
        "Temporal ordering and measurement validity must come from the research design; the software cannot infer them from a table alone.",
    ]
    return CausalResult(estimate, balance, unit, assumptions)


def bayesian_linear_regression(
    df: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    draws: int = 4000,
    prior_scale: float = 10.0,
    seed: int = 580,
) -> BayesianResult:
    """Conjugate Bayesian Gaussian regression with posterior predictive draws."""
    predictors = list(predictors)
    if not predictors:
        raise ValueError("Select at least one predictor.")
    work = df[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) <= len(predictors) + 5:
        raise ValueError("Too few complete observations for the selected Bayesian model.")
    y = work[outcome].to_numpy(float)
    Xraw = work[predictors].to_numpy(float)
    scaler = StandardScaler()
    Xstd = scaler.fit_transform(Xraw)
    X = np.column_stack([np.ones(len(work)), Xstd])
    names = ["Intercept", *predictors]
    n, p = X.shape
    tau2 = float(prior_scale) ** 2
    V0_inv = np.eye(p) / tau2
    V0_inv[0, 0] = 1e-8  # practically flat intercept prior
    Vn = np.linalg.inv(V0_inv + X.T @ X)
    bn = Vn @ (X.T @ y)
    a0, b0 = 1e-3, 1e-3
    an = a0 + n / 2
    resid_term = float(y @ y - bn @ np.linalg.solve(Vn, bn))
    bn_scale = max(1e-9, b0 + 0.5 * resid_term)
    rng = np.random.default_rng(int(seed))
    n_draws = min(max(500, int(draws)), 20000)
    sigma2 = invgamma(a=an, scale=bn_scale).rvs(size=n_draws, random_state=rng)
    beta = np.empty((n_draws, p))
    L = np.linalg.cholesky(Vn)
    for i in range(n_draws):
        beta[i] = bn + math.sqrt(float(sigma2[i])) * (L @ rng.standard_normal(p))
    q = np.quantile(beta, [0.025, .5, .975], axis=0)
    summary = pd.DataFrame({
        "term": names,
        "posterior_mean": beta.mean(axis=0),
        "posterior_sd": beta.std(axis=0, ddof=1),
        "median": q[1],
        "hdi_2.5%": q[0],
        "hdi_97.5%": q[2],
        "P(beta>0)": (beta > 0).mean(axis=0),
        "P(beta<0)": (beta < 0).mean(axis=0),
    })
    draws_df = pd.DataFrame(beta, columns=names)
    draws_df["sigma"] = np.sqrt(sigma2)
    fitted_draws = beta @ X.T
    # bounded posterior predictive sample to keep memory controlled
    eps = rng.normal(size=fitted_draws.shape) * np.sqrt(sigma2)[:, None]
    yrep = fitted_draws + eps
    pred_mean = yrep.mean(axis=0)
    pred_low, pred_high = np.quantile(yrep, [0.025, .975], axis=0)
    predictive = pd.DataFrame({"observed": y, "posterior_predictive_mean": pred_mean, "pi95_low": pred_low, "pi95_high": pred_high})
    diagnostics = pd.DataFrame([{
        "backend": "Conjugate Normal–Inverse-Gamma posterior",
        "n": n,
        "predictors": len(predictors),
        "draws": n_draws,
        "posterior_predictive_RMSE": math.sqrt(mean_squared_error(y, pred_mean)),
        "posterior_predictive_MAE": mean_absolute_error(y, pred_mean),
        "posterior_predictive_95pct_coverage": float(np.mean((y >= pred_low) & (y <= pred_high))),
        "prior_scale_standardised_coefficients": float(prior_scale),
        "seed": int(seed),
    }])
    return BayesianResult(summary, draws_df, predictive, diagnostics)


def explainable_random_forest(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    local_row: int = 0,
    seed: int = 580,
) -> ExplainableResult:
    features = list(features)
    if not features:
        raise ValueError("Select at least one feature.")
    work = df[[target, *features]].copy()
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=features + [target]).reset_index(drop=True)
    if len(work) < 30:
        raise ValueError("At least 30 complete observations are required for explainable ML.")
    X = work[features].to_numpy(float)
    y_raw = work[target]
    classification = y_raw.nunique(dropna=True) <= min(12, max(2, int(len(work) * .08)))
    if classification:
        labels = sorted(map(str, pd.unique(y_raw.astype(str))))
        mapping = {v: i for i, v in enumerate(labels)}
        y = y_raw.astype(str).map(mapping).to_numpy(int)
        model = RandomForestClassifier(n_estimators=350, min_samples_leaf=2, random_state=seed, n_jobs=-1, class_weight="balanced")
        model.fit(X, y)
        proba = model.predict_proba(X)
        pred = np.argmax(proba, axis=1)
        performance = {"task": "classification", "accuracy": float(np.mean(pred == y)), "n": len(y), "classes": len(labels)}
        if len(labels) == 2:
            performance["roc_auc_in_sample"] = float(roc_auc_score(y, proba[:, 1]))
        pred_value = [labels[i] for i in pred]
        predictions = pd.DataFrame({"observed": y_raw.astype(str), "prediction": pred_value})
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(float)
        valid = np.isfinite(y)
        X, y, work = X[valid], y[valid], work.loc[valid].reset_index(drop=True)
        model = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model.fit(X, y)
        pred = model.predict(X)
        performance = {
            "task": "regression", "n": len(y), "R2_in_sample": float(r2_score(y, pred)),
            "RMSE_in_sample": float(math.sqrt(mean_squared_error(y, pred))), "MAE_in_sample": float(mean_absolute_error(y, pred)),
        }
        predictions = pd.DataFrame({"observed": y, "prediction": pred})

    backend = "SHAP TreeExplainer"
    global_df = None
    local_df = None
    row_idx = min(max(0, int(local_row)), len(work) - 1)
    try:
        import shap  # optional accelerator
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            # positive class for binary; mean absolute across classes otherwise
            if len(vals) == 2:
                sv = np.asarray(vals[1])
            else:
                sv = np.mean(np.abs(np.stack(vals, axis=-1)), axis=-1)
        else:
            arr = np.asarray(vals)
            if arr.ndim == 3:
                sv = arr[:, :, -1] if arr.shape[-1] == 2 else np.mean(np.abs(arr), axis=-1)
            else:
                sv = arr
        global_df = pd.DataFrame({"feature": features, "mean_abs_shap": np.mean(np.abs(sv), axis=0)}).sort_values("mean_abs_shap", ascending=False)
        local_df = pd.DataFrame({"feature": features, "feature_value": X[row_idx], "shap_contribution": sv[row_idx]}).sort_values("shap_contribution", key=np.abs, ascending=False)
    except Exception:
        backend = "Permutation importance + local perturbation fallback"
        scorer = "accuracy" if classification else "neg_root_mean_squared_error"
        pi = permutation_importance(model, X, y, n_repeats=8, random_state=seed, scoring=scorer)
        global_df = pd.DataFrame({"feature": features, "mean_abs_shap": np.maximum(0.0, pi.importances_mean)}).sort_values("mean_abs_shap", ascending=False)
        baseline = X[row_idx].copy()
        base_pred = model.predict(baseline.reshape(1, -1))[0]
        rows = []
        med = np.nanmedian(X, axis=0)
        for j, f in enumerate(features):
            altered = baseline.copy(); altered[j] = med[j]
            alt_pred = model.predict(altered.reshape(1, -1))[0]
            rows.append({"feature": f, "feature_value": baseline[j], "shap_contribution": float(base_pred - alt_pred)})
        local_df = pd.DataFrame(rows).sort_values("shap_contribution", key=np.abs, ascending=False)
    return ExplainableResult(pd.DataFrame([performance]), global_df, local_df, predictions, backend)


def duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False


def pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def duckdb_query(df: pd.DataFrame, sql: str, *, max_rows: int = 100_000) -> pd.DataFrame:
    text = str(sql).strip()
    if not text:
        raise ValueError("Enter a SQL query.")
    # Read-only guard. CTEs/EXPLAIN/DESCRIBE are allowed; mutating statements are blocked.
    if re.search(r"\b(insert|update|delete|drop|alter|create|copy|attach|detach|install|load|pragma|call|export|import)\b", text, flags=re.I):
        raise ValueError("Only read-only SELECT/WITH/EXPLAIN/DESCRIBE queries are allowed in the in-app data engine.")
    try:
        import duckdb
    except Exception as exc:
        raise RuntimeError("DuckDB is not installed in this runtime. It is declared for the v5.8.0 deployment; reboot after dependencies install.") from exc
    con = duckdb.connect(database=":memory:")
    try:
        con.register("active_data", df)
        result = con.execute(text).fetchdf()
        return result.head(int(max_rows))
    finally:
        con.close()


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    except Exception as exc:
        raise RuntimeError("PyArrow is required for Parquet export and is declared for the v5.8.0 deployment.") from exc
    return buffer.getvalue()


def arrow_ipc_bytes(df: pd.DataFrame) -> bytes:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as exc:
        raise RuntimeError("PyArrow is required for Arrow IPC export and is declared for the v5.8.0 deployment.") from exc
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()
