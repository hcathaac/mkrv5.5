"""Auditable multi-criteria decision analysis for the v5.2.1 dashboard.

The engine is deliberately independent of Streamlit.  It exposes every
transformation, weight and rank so the results can be inspected, exported and
reproduced rather than treated as an opaque decision score.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import zipfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from output_guidance import build_output_guide, output_guide_markdown
from scipy import stats


@dataclass
class MCDAOutput:
    rankings: pd.DataFrame
    weights: pd.DataFrame
    normalised_matrix: pd.DataFrame
    rank_correlations: pd.DataFrame
    sensitivity: pd.DataFrame
    acceptability_summary: pd.DataFrame
    rank_acceptability: pd.DataFrame
    diagnostics: pd.DataFrame
    interpretation: list[str]
    primary_method: str


METHODS = ("MAVT", "TOPSIS", "PROMETHEE II")
WEIGHT_METHODS = ("Equal", "User-defined", "Entropy", "CRITIC", "AHP pairwise")


def _normalise_weights(values: Sequence[float]) -> np.ndarray:
    w = np.asarray(values, dtype=float)
    if w.ndim != 1 or not len(w) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Criterion weights must be finite, non-negative numbers.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("At least one criterion weight must be positive.")
    return w / total


def ahp_weights(pairwise: pd.DataFrame | np.ndarray, criteria: Sequence[str]) -> tuple[np.ndarray, float, float]:
    """Return principal-eigenvector AHP weights, lambda-max and consistency ratio.

    Values above the diagonal are treated as authoritative.  The reciprocal
    lower triangle is rebuilt, which prevents internally contradictory pairs.
    """
    names = list(criteria)
    matrix = np.asarray(pairwise, dtype=float).copy()
    n = len(names)
    if matrix.shape != (n, n):
        raise ValueError(f"The AHP matrix must be {n} x {n}.")
    if np.any(~np.isfinite(matrix)) or np.any(matrix <= 0):
        raise ValueError("All AHP comparisons must be finite and strictly positive.")
    np.fill_diagonal(matrix, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(matrix[i, j])
            matrix[j, i] = 1.0 / value
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = int(np.argmax(eigenvalues.real))
    principal = np.abs(eigenvectors[:, idx].real)
    weights = _normalise_weights(principal)
    lambda_max = float(eigenvalues[idx].real)
    ci = max((lambda_max - n) / (n - 1), 0.0) if n > 1 else 0.0
    random_index = {1: 0.0, 2: 0.0, 3: .58, 4: .90, 5: 1.12, 6: 1.24,
                    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51,
                    12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}.get(n, 1.59)
    cr = float(ci / random_index) if random_index else 0.0
    return weights, lambda_max, cr


def _prepare_matrix(
    df: pd.DataFrame,
    criteria: Sequence[str],
    directions: Mapping[str, str],
    alternative_id: str | None,
    missing: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    columns = [c for c in dict.fromkeys(criteria) if c in df.columns]
    if len(columns) < 2:
        raise ValueError("Select at least two numeric decision criteria.")
    if len(columns) > 50:
        raise ValueError("The dedicated MCDA engine supports a maximum of 50 criteria per run.")
    raw = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if missing == "Complete cases":
        keep = raw.notna().all(axis=1)
        raw = raw.loc[keep]
    else:
        usable = raw.notna().any(axis=0)
        raw = raw.loc[:, usable]
        columns = list(raw.columns)
        raw = raw.fillna(raw.median(numeric_only=True))
        keep = raw.notna().all(axis=1)
        raw = raw.loc[keep]
    if len(raw) < 2:
        raise ValueError("Fewer than two complete alternatives remain after missing-data handling.")
    if len(columns) < 2:
        raise ValueError("Fewer than two usable criteria remain after missing-data handling.")

    if alternative_id and alternative_id in df.columns:
        ids = df.loc[raw.index, alternative_id].astype(str).fillna("Missing identifier")
    else:
        ids = pd.Series([f"Alternative {i + 1}" for i in range(len(raw))], index=raw.index)
    duplicates = ids.duplicated(keep=False)
    if duplicates.any():
        occurrence = ids.groupby(ids).cumcount().add(1).astype(str)
        ids = ids.where(~duplicates, ids + " [" + occurrence + "]")

    oriented = pd.DataFrame(index=raw.index)
    constant: list[str] = []
    for col in columns:
        values = raw[col].to_numpy(float)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(high, low):
            oriented[col] = .5
            constant.append(col)
            continue
        if str(directions.get(col, "Maximise")).lower().startswith("min"):
            oriented[col] = (high - values) / (high - low)
        else:
            oriented[col] = (values - low) / (high - low)
    oriented.insert(0, "alternative", ids.to_numpy())
    raw_export = raw.copy()
    raw_export.insert(0, "alternative", ids.to_numpy())
    return raw_export.reset_index(drop=True), oriented.reset_index(drop=True), columns, constant


def _entropy_weights(z: np.ndarray) -> np.ndarray:
    n = z.shape[0]
    shifted = np.clip(z, 0, None) + 1e-12
    proportions = shifted / shifted.sum(axis=0, keepdims=True)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / np.log(max(n, 2))
    diversification = np.clip(1 - entropy, 0, None)
    return _normalise_weights(diversification if diversification.sum() > 1e-12 else np.ones(z.shape[1]))


def _critic_weights(z: np.ndarray) -> np.ndarray:
    spread = np.std(z, axis=0, ddof=1)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=1.0)
    information = spread * np.sum(1 - corr, axis=1)
    return _normalise_weights(np.clip(information, 0, None) if information.sum() > 1e-12 else np.ones(z.shape[1]))


def _promethee_net_flow(z: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(z)
    leaving = np.zeros(n, dtype=float)
    entering = np.zeros(n, dtype=float)
    denominator = max(n - 1, 1)
    for j, weight in enumerate(weights):
        diff = z[:, j, None] - z[None, :, j]
        preference = np.maximum(diff, 0.0)
        leaving += weight * preference.sum(axis=1) / denominator
        entering += weight * preference.sum(axis=0) / denominator
    return leaving - entering, leaving, entering


def _method_scores(z: np.ndarray, weights: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if method == "TOPSIS":
        weighted = z * weights
        positive = weights
        negative = np.zeros_like(weights)
        d_positive = np.sqrt(((weighted - positive) ** 2).sum(axis=1))
        d_negative = np.sqrt(((weighted - negative) ** 2).sum(axis=1))
        denominator = d_positive + d_negative
        score = np.divide(d_negative, denominator, out=np.full_like(d_negative, .5), where=denominator > 0)
        return score, {"distance_to_ideal": d_positive, "distance_to_anti_ideal": d_negative}
    if method == "PROMETHEE II":
        net, leaving, entering = _promethee_net_flow(z, weights)
        return net, {"leaving_flow": leaving, "entering_flow": entering}
    score = z @ weights
    return score, {"additive_value": score}


def _rank_descending(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", ascending=False).astype(int).to_numpy()


def _perturb_weight(weights: np.ndarray, criterion: int, target: float) -> np.ndarray:
    result = weights.copy()
    target = float(np.clip(target, 0, 1))
    others = np.arange(len(weights)) != criterion
    remaining = 1 - target
    other_sum = float(weights[others].sum())
    result[criterion] = target
    result[others] = weights[others] / other_sum * remaining if other_sum > 0 else remaining / max(others.sum(), 1)
    return _normalise_weights(result)


def _weight_sensitivity(
    z: np.ndarray,
    ids: Sequence[str],
    criteria: Sequence[str],
    weights: np.ndarray,
    method: str,
    variation: float,
) -> pd.DataFrame:
    base_scores, _ = _method_scores(z, weights, method)
    base_rank = _rank_descending(base_scores)
    base_top = str(ids[int(np.argmax(base_scores))])
    rows: list[dict] = []
    for j, criterion in enumerate(criteria):
        for scenario, factor in (("Lower", 1 - variation), ("Higher", 1 + variation)):
            changed = _perturb_weight(weights, j, weights[j] * factor)
            score, _ = _method_scores(z, changed, method)
            rank = _rank_descending(score)
            rho = float(stats.spearmanr(base_rank, rank).statistic)
            top = str(ids[int(np.argmax(score))])
            rows.append({
                "criterion": criterion,
                "scenario": scenario,
                "base_weight": weights[j],
                "perturbed_weight": changed[j],
                "spearman_rank_stability": rho,
                "top_alternative": top,
                "top_unchanged": top == base_top,
            })
    return pd.DataFrame(rows)


def _monte_carlo_acceptability(
    z: np.ndarray,
    ids: Sequence[str],
    weights: np.ndarray,
    method: str,
    simulations: int,
    concentration: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method not in {"MAVT", "TOPSIS"}:
        raise ValueError("Monte Carlo rank acceptability is available for MAVT and TOPSIS.")
    simulations = int(np.clip(simulations, 100, 10_000))
    rng = np.random.default_rng(seed)
    alpha = np.clip(weights * float(concentration), .05, None)
    sampled_weights = rng.dirichlet(alpha, size=simulations)
    n = len(z)
    rank_limit = min(20, n)
    counts = np.zeros((n, rank_limit), dtype=np.int64)
    rank_sum = np.zeros(n, dtype=float)
    rank_sq_sum = np.zeros(n, dtype=float)
    top3 = np.zeros(n, dtype=np.int64)
    for sampled in sampled_weights:
        score, _ = _method_scores(z, sampled, method)
        order = np.argsort(-score, kind="stable")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        rank_sum += ranks
        rank_sq_sum += ranks ** 2
        top3[order[:min(3, n)]] += 1
        for position, alternative in enumerate(order[:rank_limit]):
            counts[alternative, position] += 1
    expected = rank_sum / simulations
    variance = np.maximum(rank_sq_sum / simulations - expected ** 2, 0)
    summary = pd.DataFrame({
        "alternative": list(ids),
        "probability_rank_1": counts[:, 0] / simulations,
        "probability_top_3": top3 / simulations,
        "expected_rank": expected,
        "rank_standard_deviation": np.sqrt(variance),
    }).sort_values(["probability_rank_1", "expected_rank"], ascending=[False, True]).reset_index(drop=True)
    accept = pd.DataFrame(counts / simulations, columns=[f"rank_{i}" for i in range(1, rank_limit + 1)])
    accept.insert(0, "alternative", list(ids))
    return summary, accept


def mcda_analysis(
    df: pd.DataFrame,
    criteria: Sequence[str],
    directions: Mapping[str, str] | None = None,
    weight_method: str = "Equal",
    user_weights: Mapping[str, float] | None = None,
    pairwise_matrix: pd.DataFrame | np.ndarray | None = None,
    methods: Sequence[str] = METHODS,
    alternative_id: str | None = None,
    missing: str = "Median imputation",
    primary_method: str = "TOPSIS",
    simulations: int = 1_000,
    concentration: float = 75.0,
    sensitivity_range: float = .25,
    seed: int = 42,
) -> MCDAOutput:
    """Run auditable MAVT, TOPSIS and PROMETHEE-II ranking with robustness checks."""
    directions = directions or {c: "Maximise" for c in criteria}
    raw, normalised, columns, constant = _prepare_matrix(df, criteria, directions, alternative_id, missing)
    z = normalised[columns].to_numpy(float)
    ids = normalised["alternative"].astype(str).tolist()
    chosen_methods = [m for m in dict.fromkeys(methods) if m in METHODS]
    if not chosen_methods:
        raise ValueError("Select at least one MCDA ranking method.")
    if primary_method not in {"MAVT", "TOPSIS"}:
        primary_method = "TOPSIS" if "TOPSIS" in chosen_methods else "MAVT"
    if primary_method not in chosen_methods:
        chosen_methods.insert(0, primary_method)

    ahp_lambda = np.nan
    ahp_cr = np.nan
    if weight_method == "Entropy":
        weights = _entropy_weights(z)
    elif weight_method == "CRITIC":
        weights = _critic_weights(z)
    elif weight_method == "AHP pairwise":
        if pairwise_matrix is None:
            raise ValueError("Provide the AHP pairwise comparison matrix.")
        weights, ahp_lambda, ahp_cr = ahp_weights(pairwise_matrix, columns)
    elif weight_method == "User-defined":
        supplied = user_weights or {}
        weights = _normalise_weights([float(supplied.get(c, 0)) for c in columns])
    else:
        weights = np.repeat(1 / len(columns), len(columns))

    weight_table = pd.DataFrame({
        "criterion": columns,
        "direction": [directions.get(c, "Maximise") for c in columns],
        "weight": weights,
        "constant_criterion": [c in constant for c in columns],
    })
    rankings = pd.DataFrame({"alternative": ids})
    rank_columns: list[str] = []
    for method in chosen_methods:
        score, extras = _method_scores(z, weights, method)
        rankings[f"{method}_score"] = score
        rankings[f"{method}_rank"] = _rank_descending(score)
        rank_columns.append(f"{method}_rank")
        for name, values in extras.items():
            if name not in rankings:
                rankings[f"{method}_{name}"] = values
    if len(rank_columns) > 1:
        n = len(rankings)
        percentile = pd.concat([(n - rankings[c]) / max(n - 1, 1) for c in rank_columns], axis=1)
        rankings["Consensus_score"] = percentile.mean(axis=1)
        rankings["Consensus_rank"] = _rank_descending(rankings["Consensus_score"].to_numpy(float))
    rankings = rankings.sort_values(f"{primary_method}_rank").reset_index(drop=True)

    corr = rankings[rank_columns].corr(method="spearman")
    corr.index.name = "method"
    correlations = corr.reset_index()
    sensitivity = _weight_sensitivity(z, ids, columns, weights, primary_method, float(sensitivity_range))
    accept_summary, rank_accept = _monte_carlo_acceptability(
        z, ids, weights, primary_method, simulations, concentration, seed,
    )
    diagnostics = pd.DataFrame([{
        "alternatives": len(rankings),
        "criteria": len(columns),
        "missing_data_rule": missing,
        "weight_method": weight_method,
        "primary_method": primary_method,
        "monte_carlo_simulations": int(np.clip(simulations, 100, 10_000)),
        "dirichlet_concentration": float(concentration),
        "sensitivity_range": float(sensitivity_range),
        "ahp_lambda_max": ahp_lambda,
        "ahp_consistency_ratio": ahp_cr,
        "constant_criteria": ", ".join(constant) if constant else "None",
    }])
    top = rankings.iloc[0]["alternative"]
    stable_share = float(sensitivity.top_unchanged.mean()) if len(sensitivity) else np.nan
    interpretation = [
        f"{top} is ranked first under the selected primary method ({primary_method}); this is a conditional decision ranking, not a causal estimate.",
        f"The top alternative remains unchanged in {stable_share:.1%} of the one-at-a-time weight perturbation scenarios.",
        "Monte Carlo rank acceptability varies criterion weights around the stated baseline; probabilities quantify ranking robustness, not the probability that a project will succeed.",
        "Compare MAVT, TOPSIS and PROMETHEE-II ranks. Large disagreements indicate compensability or preference-structure sensitivity that should be reported rather than averaged away.",
    ]
    if weight_method == "AHP pairwise":
        interpretation.append(
            f"The AHP consistency ratio is {ahp_cr:.3f}. Values above 0.10 normally require reconsideration of the pairwise judgements."
        )
    if constant:
        interpretation.append("Constant criteria carry no discriminatory information and are flagged in the weight table: " + ", ".join(constant) + ".")
    normalised_export = normalised.copy()
    return MCDAOutput(
        rankings=rankings,
        weights=weight_table,
        normalised_matrix=normalised_export,
        rank_correlations=correlations,
        sensitivity=sensitivity,
        acceptability_summary=accept_summary,
        rank_acceptability=rank_accept,
        diagnostics=diagnostics,
        interpretation=interpretation,
        primary_method=primary_method,
    )


def _save_figure_set(zf: zipfile.ZipFile, fig, stem: str) -> None:
    for fmt in ("png", "svg", "pdf"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=600 if fmt == "png" else None,
                    bbox_inches="tight", facecolor="white")
        zf.writestr(f"{stem}.{fmt}", buffer.getvalue())


def mcda_publication_bundle(output: MCDAOutput) -> bytes:
    """Create colour and black-and-white 600-dpi/vector MCDA packages."""
    from matplotlib import pyplot as plt

    archive = io.BytesIO()
    method = output.primary_method
    score_col = f"{method}_score"
    rank_col = f"{method}_rank"
    top = output.rankings.nsmallest(min(25, len(output.rankings)), rank_col).sort_values(rank_col, ascending=False)
    weights = output.weights.sort_values("weight")
    top_ids = output.rankings.nsmallest(min(20, len(output.rankings)), rank_col).alternative.astype(str)
    matrix = output.normalised_matrix.set_index("alternative").reindex(top_ids)
    matrix = matrix[[c for c in output.weights.criterion if c in matrix]][:20]
    accept = output.rank_acceptability.set_index("alternative").reindex(top_ids)
    accept = accept.iloc[:, :min(10, accept.shape[1])]

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            main = "#4D4D4D" if monochrome else "#155B8A"
            accent = "#111111" if monochrome else "#D89B2B"
            fig, ax = plt.subplots(figsize=(8.27, max(5.3, .25 * len(top))), constrained_layout=True)
            ax.barh(top.alternative.astype(str), top[score_col], color=main, edgecolor="#222222",
                    hatch="//" if monochrome else "")
            ax.set_xlabel(f"{method} score"); ax.set_title(f"MCDA ranking: {method}", loc="left", fontsize=15, fontweight="bold")
            ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_mcda_ranking")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.27, max(4.8, .28 * len(weights))), constrained_layout=True)
            ax.barh(weights.criterion.astype(str), weights.weight, color=accent, edgecolor="#222222",
                    hatch="//" if monochrome else "")
            ax.set_xlabel("Normalised criterion weight"); ax.set_title("MCDA criterion weights", loc="left", fontsize=15, fontweight="bold")
            ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_criterion_weights")
            plt.close(fig)

            if not matrix.empty:
                fig, ax = plt.subplots(figsize=(max(8.27, .42 * matrix.shape[1]), max(5.5, .32 * matrix.shape[0])), constrained_layout=True)
                image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Greys" if monochrome else "viridis", vmin=0, vmax=1)
                ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns.astype(str), rotation=45, ha="right")
                ax.set_yticks(np.arange(matrix.shape[0]), matrix.index.astype(str))
                ax.set_title("Oriented criterion performance of leading alternatives", loc="left", fontsize=15, fontweight="bold")
                fig.colorbar(image, ax=ax, label="Normalised performance", shrink=.82)
                _save_figure_set(zf, fig, f"{style}_criterion_performance")
                plt.close(fig)

            if not accept.empty:
                fig, ax = plt.subplots(figsize=(8.27, max(5.2, .32 * accept.shape[0])), constrained_layout=True)
                image = ax.imshow(accept.to_numpy(float), aspect="auto", cmap="Greys" if monochrome else "magma", vmin=0, vmax=1)
                ax.set_xticks(np.arange(accept.shape[1]), [c.replace("rank_", "Rank ") for c in accept.columns], rotation=45, ha="right")
                ax.set_yticks(np.arange(accept.shape[0]), accept.index.astype(str))
                ax.set_title("Monte Carlo rank acceptability", loc="left", fontsize=15, fontweight="bold")
                fig.colorbar(image, ax=ax, label="Probability", shrink=.82)
                _save_figure_set(zf, fig, f"{style}_rank_acceptability")
                plt.close(fig)

        tables = {
            "mcda_rankings.csv": output.rankings,
            "criterion_weights.csv": output.weights,
            "normalised_decision_matrix.csv": output.normalised_matrix,
            "method_rank_correlations.csv": output.rank_correlations,
            "weight_sensitivity.csv": output.sensitivity,
            "rank_acceptability_summary.csv": output.acceptability_summary,
            "rank_acceptability_matrix.csv": output.rank_acceptability,
            "mcda_diagnostics.csv": output.diagnostics,
        }
        for name, table in tables.items():
            zf.writestr(name, table.to_csv(index=False).encode("utf-8-sig"))
        guide = build_output_guide(
            {
                "MCDA rankings": output.rankings,
                "Criterion weights": output.weights,
                "Normalised decision matrix": output.normalised_matrix,
                "Method rank correlations": output.rank_correlations,
                "Weight sensitivity": output.sensitivity,
                "Rank acceptability summary": output.acceptability_summary,
                "Rank acceptability matrix": output.rank_acceptability,
                "MCDA diagnostics": output.diagnostics,
            },
            ["MCDA ranking", "MCDA criterion weights", "Criterion performance heatmap", "Monte Carlo rank acceptability heatmap"],
        )
        zf.writestr("output_interpretation_guide.csv", guide.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("OUTPUT_INTERPRETATION_GUIDE.md", output_guide_markdown(guide))
        zf.writestr(
            "README.txt",
            ("Dedicated MCDA Engine v5.2.1\n"
             "Methods: MAVT, TOPSIS and PROMETHEE II; equal/user/entropy/CRITIC/AHP weights; "
             "one-at-a-time sensitivity and Monte Carlo rank acceptability.\n"
             "Figures: colour and black-and-white; PNG is 600 dpi; SVG/PDF are vector formats.\n"
             "Rankings depend on criteria, directions, scaling, weights, missing-data rules and the selected preference model.\n").encode("utf-8"),
        )
    return archive.getvalue()
