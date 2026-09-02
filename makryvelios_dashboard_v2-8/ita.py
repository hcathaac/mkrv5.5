"""Iterative Trichotomic Approach engines for public-funding portfolios.

The implementation follows the published score-uncertainty and converging-
weights ITA logic, then exposes the policy/equity and hybrid extensions as
auditable Python models.  SciPy/HiGHS executes the binary optimisation; every
run can also be exported as a self-contained GAMS model and data package.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


CALL_BUDGETS = {
    "AT01": 300_000_000.0,
    "AT02": 200_000_000.0,
    "AT03": 150_000_000.0,
    "AT04": 220_000_000.0,
    "AT05": 150_000_000.0,
    "AT06": 400_000_000.0,
    "AT07": 200_000_000.0,
    "AT08": 130_000_000.0,
    "AT09": 100_000_000.0,
    "AT10": 40_000_000.0,
    "AT11": 40_000_000.0,
    "AT12": 120_000_000.0,
    "AT14": 50_000_000.0,
}

BENEFICIARY_CATEGORY_CAPS = {
    "M1": 40_000_000.0,
    "M2": 25_000_000.0,
    "M3": 22_000_000.0,
    "M4": 18_000_000.0,
    "M5": 12_000_000.0,
    "M6": 6_000_000.0,
    "M7": 30_000_000.0,
}

DEFAULT_CRITERION_WEIGHTS = {
    "C1": 0.25,
    "C2": 0.20,
    "C3": 0.20,
    "C4": 0.15,
    "C5": 0.15,
    "C6": 0.05,
}


@dataclass
class ITAOutput:
    variant: str
    projects: pd.DataFrame
    rounds: pd.DataFrame
    inclusion_history: pd.DataFrame
    weights_history: pd.DataFrame
    portfolio_summary: pd.DataFrame
    call_allocation: pd.DataFrame
    beneficiary_allocation: pd.DataFrame
    regional_allocation: pd.DataFrame
    scorecards: pd.DataFrame
    diagnostics: pd.DataFrame
    interpretation: list[str]
    settings: dict


def normalise_call_code(value: object) -> str:
    text = str(value).strip().upper().replace("Α", "A").replace("Τ", "T")
    digits = re.sub(r"\D", "", text)
    return f"AT{int(digits):02d}" if digits else text


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "pass", "eligible", "selected", "ναι", "επιλέξιμο", "επιλεγμένο"})


def _normalise_weights(weights: Sequence[float]) -> np.ndarray:
    out = np.asarray(weights, dtype=float)
    if out.ndim != 1 or not np.isfinite(out).all() or (out < 0).any() or out.sum() <= 0:
        raise ValueError("Criterion weights must be finite, non-negative and have a positive sum.")
    return out / out.sum()


def _score_to_ten(series: pd.Series, scale_scores: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Criterion '{series.name}' contains no numeric values.")
    values = values.fillna(values.median())
    if not scale_scores:
        if ((values < 0) | (values > 10)).any():
            raise ValueError(f"Criterion '{series.name}' lies outside 0-10. Enable score scaling or correct the data.")
        return values.astype(float)
    low, high = float(values.min()), float(values.max())
    if low >= 0 and high <= 1:
        return values.astype(float) * 10.0
    if low >= 0 and high <= 10:
        return values.astype(float)
    if math.isclose(low, high):
        return pd.Series(np.full(len(values), 5.0), index=values.index, name=series.name)
    return ((values - low) / (high - low) * 10.0).astype(float)


def prepare_ita_projects(
    data: pd.DataFrame,
    *,
    project_id: str,
    call: str,
    beneficiary: str,
    region: str | None,
    requested_budget: str,
    criteria: Sequence[str],
    weights: Sequence[float],
    eligibility_columns: Sequence[str] = (),
    disadvantaged_column: str | None = None,
    beneficiary_category: str | None = None,
    actual_selected: str | None = None,
    geography_cost_factor: str | None = None,
    scale_scores: bool = False,
    disadvantaged_c1_threshold: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a validated, model-ready project table without mutating input data."""
    required = [project_id, call, beneficiary, requested_budget, *criteria]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(criteria) < 2:
        raise ValueError("Select at least two evaluation criteria.")
    w = _normalise_weights(weights)
    budget = pd.to_numeric(data[requested_budget], errors="coerce")
    valid_budget = budget.gt(0) & np.isfinite(budget)
    out = pd.DataFrame(index=data.index)
    raw_ids = data[project_id].astype(str).str.strip().replace({"": "unnamed"})
    duplicates = raw_ids.groupby(raw_ids).cumcount()
    out["project_id"] = np.where(duplicates.eq(0), raw_ids, raw_ids + " [" + (duplicates + 1).astype(str) + "]")
    out["call"] = data[call].map(normalise_call_code)
    out["beneficiary"] = data[beneficiary].astype(str).str.strip().replace({"": "Unspecified"})
    out["region"] = data[region].astype(str).str.strip().replace({"": "Unspecified"}) if region else "Unspecified"
    out["requested_budget"] = budget.astype(float)
    out["eligible"] = valid_budget
    for column in eligibility_columns:
        if column in data:
            out["eligible"] &= _truthy(data[column])
    if beneficiary_category and beneficiary_category in data:
        out["beneficiary_category"] = data[beneficiary_category].astype(str).str.strip().str.upper()
    else:
        out["beneficiary_category"] = ""
    if actual_selected and actual_selected in data:
        out["actual_selected"] = _truthy(data[actual_selected]).astype(int)
    else:
        out["actual_selected"] = np.nan
    if geography_cost_factor and geography_cost_factor in data:
        factor = pd.to_numeric(data[geography_cost_factor], errors="coerce").fillna(1.0).clip(lower=0.01)
    else:
        factor = pd.Series(1.0, index=data.index)
    out["geography_cost_factor"] = factor.astype(float)

    criterion_rows: list[dict] = []
    for position, (column, weight) in enumerate(zip(criteria, w), start=1):
        code = f"C{position}"
        out[code] = _score_to_ten(data[column], scale_scores)
        criterion_rows.append({"criterion": code, "source_column": column, "weight": float(weight)})
    criterion_codes = [row["criterion"] for row in criterion_rows]
    out["final_score"] = out[criterion_codes].to_numpy(float) @ w
    out["rank_within_call"] = out.groupby("call")["final_score"].rank(method="min", ascending=False).astype(int)
    if disadvantaged_column and disadvantaged_column in data:
        out["disadvantaged"] = _truthy(data[disadvantaged_column])
    else:
        out["disadvantaged"] = out["C1"].ge(float(disadvantaged_c1_threshold))
    out["source_row"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True), pd.DataFrame(criterion_rows)


def _beneficiary_cap_map(projects: pd.DataFrame, category_caps: Mapping[str, float] | None) -> dict[str, float]:
    if not category_caps:
        return {}
    caps = {str(k).strip().upper(): float(v) for k, v in category_caps.items() if float(v) > 0}
    result: dict[str, float] = {}
    for beneficiary, group in projects.groupby("beneficiary", sort=False):
        categories = group["beneficiary_category"].dropna().astype(str).str.strip().str.upper()
        available = [caps[c] for c in categories if c in caps]
        if available:
            result[str(beneficiary)] = min(available)
    return result


def solve_portfolio(
    projects: pd.DataFrame,
    utility: Sequence[float],
    *,
    call_budgets: Mapping[str, float],
    beneficiary_caps: Mapping[str, float] | None = None,
    costs: Sequence[float] | None = None,
    fixed_in: Sequence[int] = (),
    fixed_out: Sequence[int] = (),
    equity_floor: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Solve the binary public-funding portfolio exactly with HiGHS MILP."""
    n = len(projects)
    if n == 0:
        raise ValueError("No projects are available for optimisation.")
    values = np.asarray(utility, dtype=float)
    expense = np.asarray(projects["requested_budget"] if costs is None else costs, dtype=float)
    if len(values) != n or len(expense) != n or not np.isfinite(values).all() or not np.isfinite(expense).all():
        raise ValueError("Utility and project-cost vectors must be complete and finite.")
    if (expense <= 0).any():
        raise ValueError("Every optimised project must have a positive requested budget.")
    budgets = {normalise_call_code(k): float(v) for k, v in call_budgets.items() if float(v) >= 0}
    observed_calls = set(projects.loc[projects["eligible"], "call"].astype(str))
    missing_calls = sorted(observed_calls - set(budgets))
    if missing_calls:
        raise ValueError("No positive funding envelope was supplied for: " + ", ".join(missing_calls))

    rows: list[np.ndarray] = []
    upper: list[float] = []
    lower: list[float] = []
    constraint_names: list[str] = []
    calls = projects["call"].astype(str).to_numpy()
    beneficiaries = projects["beneficiary"].astype(str).to_numpy()
    for code in sorted(observed_calls):
        row = np.where(calls == code, expense, 0.0)
        rows.append(row); lower.append(-np.inf); upper.append(budgets[code]); constraint_names.append(f"call:{code}")
    for name, cap in sorted((beneficiary_caps or {}).items()):
        row = np.where(beneficiaries == str(name), expense, 0.0)
        if row.any() and float(cap) > 0:
            rows.append(row); lower.append(-np.inf); upper.append(float(cap)); constraint_names.append(f"beneficiary:{name}")
    if equity_floor > 0:
        disadvantaged = projects["disadvantaged"].astype(bool).to_numpy()
        row = float(equity_floor) * expense - np.where(disadvantaged, expense, 0.0)
        rows.append(row); lower.append(-np.inf); upper.append(0.0); constraint_names.append("equity_floor")

    lb = np.zeros(n)
    ub = np.where(projects["eligible"].astype(bool).to_numpy(), 1.0, 0.0)
    for index in fixed_in:
        lb[int(index)] = 1.0; ub[int(index)] = 1.0
    for index in fixed_out:
        lb[int(index)] = 0.0; ub[int(index)] = 0.0
    matrix = csr_matrix(np.vstack(rows)) if rows else csr_matrix((0, n))
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)) if rows else None
    tie_break = 1e-8 * projects["C1"].to_numpy(float) - 1e-14 * expense
    result = milp(
        c=-(values + tie_break),
        integrality=np.ones(n),
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise ValueError(f"Portfolio optimisation failed: {result.message}")
    selected = (result.x >= 0.5).astype(int)
    selected_cost = float(np.dot(expense, selected))
    disadvantaged_cost = float(np.dot(expense * projects["disadvantaged"].astype(int).to_numpy(), selected))
    return selected, {
        "solver": "SciPy HiGHS MILP",
        "status": str(result.message),
        "objective": float(np.dot(values, selected)),
        "selected_projects": int(selected.sum()),
        "allocated_budget": selected_cost,
        "equity_index": disadvantaged_cost / selected_cost if selected_cost else np.nan,
        "constraints": len(constraint_names),
        "constraint_names": constraint_names,
    }


def converging_weight_sets(final_weights: Sequence[float], round_number: int, rounds: int) -> np.ndarray:
    """Exact ITA-II convergence from one-criterion vectors to final weights."""
    weights = _normalise_weights(final_weights)
    if rounds < 2 or round_number < 1 or round_number > rounds:
        raise ValueError("The round number must lie within an ITA design of at least two rounds.")
    alpha = (round_number - 1) / (rounds - 1)
    result = np.tile(alpha * weights, (len(weights), 1))
    result[np.arange(len(weights)), np.arange(len(weights))] += 1.0 - alpha
    return result


def _classify_probability(probability: np.ndarray, green_threshold: float, red_threshold: float) -> np.ndarray:
    return np.where(probability >= green_threshold, "Green", np.where(probability <= red_threshold, "Red", "Gray"))


def _allocation_tables(projects: pd.DataFrame, selected_column: str, cost_column: str = "allocated_budget") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen = projects.loc[projects[selected_column].eq(1)].copy()
    def grouped(column: str) -> pd.DataFrame:
        if chosen.empty:
            return pd.DataFrame(columns=[column, "selected_projects", "allocated_budget", "mean_score"])
        return chosen.groupby(column, as_index=False).agg(
            selected_projects=("project_id", "size"),
            allocated_budget=(cost_column, "sum"),
            mean_score=("final_score", "mean"),
        ).sort_values("allocated_budget", ascending=False)
    return grouped("call"), grouped("beneficiary"), grouped("region")


def _scorecard_table(projects: pd.DataFrame, decision_column: str, classification_column: str) -> pd.DataFrame:
    records: list[dict] = []
    criterion_columns = [c for c in projects.columns if re.fullmatch(r"C\d+", str(c))]
    for _, row in projects.iterrows():
        ranked = sorted(((c, float(row[c])) for c in criterion_columns), key=lambda item: item[1], reverse=True)
        strongest = ", ".join(f"{c} ({v:.1f})" for c, v in ranked[:2])
        weakest = ", ".join(f"{c} ({v:.1f})" for c, v in ranked[-2:])
        decision = "selected" if int(row[decision_column]) == 1 else "not selected"
        sentences = [
            f"Project {row.project_id} is {decision} in the final portfolio for {row.call} and is ranked {int(row.rank_within_call)} within that call.",
            f"Its weighted score is {row.final_score:.2f}/10; the strongest criterion evidence is {strongest}, while the weakest is {weakest}.",
            f"The requested budget is EUR {row.requested_budget:,.0f} and the assigned certainty class is {row[classification_column]}.",
            ("The project contributes to the disadvantaged-area funding share used by the equity constraint." if bool(row.disadvantaged)
             else "The project is outside the disadvantaged-area group under the current policy definition."),
        ]
        records.append({
            "project_id": row.project_id,
            "call": row.call,
            "beneficiary": row.beneficiary,
            "region": row.region,
            "final_score": row.final_score,
            "rank_within_call": row.rank_within_call,
            "requested_budget": row.requested_budget,
            "decision": decision,
            "classification": row[classification_column],
            "explanation": " ".join(sentences),
        })
    return pd.DataFrame(records)


def run_policy_ita(
    projects: pd.DataFrame,
    *,
    call_budgets: Mapping[str, float],
    beneficiary_category_caps: Mapping[str, float] | None = None,
    policy_strength: float = 0.25,
    equity_floor: float = 0.15,
) -> ITAOutput:
    """Run the four-round Policy-Based ITA extension."""
    if not 0 <= policy_strength <= 1 or not 0 <= equity_floor < 1:
        raise ValueError("Policy strength and equity floor must be proportions between zero and one.")
    caps = _beneficiary_cap_map(projects, beneficiary_category_caps)
    base = projects["final_score"].to_numpy(float)
    c1 = projects["C1"].to_numpy(float)
    designs = [
        (1, "Pure score ranking", 0.0, 0.0, False),
        (2, "C1 policy priority", policy_strength, 0.0, False),
        (3, "Equity floor", policy_strength, equity_floor, False),
        (4, "Full policy optimisation", policy_strength, equity_floor, True),
    ]
    histories: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for round_number, label, strength, floor, use_caps in designs:
        utility = (1.0 - strength) * base + strength * c1
        selected, meta = solve_portfolio(
            projects, utility, call_budgets=call_budgets,
            beneficiary_caps=caps if use_caps else {}, equity_floor=floor,
        )
        histories.append(pd.DataFrame({"project_id": projects.project_id, "round": round_number, "round_label": label, "selected": selected}))
        summaries.append({"round": round_number, "round_label": label, "policy_strength": strength, "equity_floor": floor, "beneficiary_caps_active": use_caps, **{k: v for k, v in meta.items() if k != "constraint_names"}})
    history = pd.concat(histories, ignore_index=True)
    pivot = history.pivot(index="project_id", columns="round", values="selected").reindex(projects.project_id).fillna(0).astype(int)
    final = pivot[4].to_numpy()
    count = pivot.sum(axis=1).to_numpy()
    classification = np.select(
        [count == 4, count == 0, (pivot[1].to_numpy() == 0) & (final == 1), (pivot[1].to_numpy() == 1) & (final == 0)],
        ["Policy-robust green", "Policy-robust red", "Equity-sensitive gain", "Equity-sensitive loss"],
        default="Policy-conflict zone",
    )
    result = projects.copy()
    result["conventional_selected"] = pivot[1].to_numpy()
    result["policy_selected"] = final
    result["policy_rounds_selected"] = count
    result["policy_classification"] = classification
    result["allocated_budget"] = result["requested_budget"] * result["policy_selected"]
    call, beneficiary, regional = _allocation_tables(result, "policy_selected")
    scorecards = _scorecard_table(result, "policy_selected", "policy_classification")
    final_meta = summaries[-1]
    unallocated = sum(float(v) for v in call_budgets.values()) - float(result.allocated_budget.sum())
    portfolio_summary = pd.DataFrame([{**final_meta, "unallocated_budget": unallocated}])
    diagnostics = pd.DataFrame([
        {"check": "Eligible projects", "value": int(result.eligible.sum())},
        {"check": "Ineligible projects", "value": int((~result.eligible).sum())},
        {"check": "Policy-robust green", "value": int((classification == "Policy-robust green").sum())},
        {"check": "Policy-conflict/equity-sensitive", "value": int(np.isin(classification, ["Policy-conflict zone", "Equity-sensitive gain", "Equity-sensitive loss"]).sum())},
    ])
    return ITAOutput(
        variant="ITA-PB", projects=result, rounds=pd.DataFrame(summaries), inclusion_history=history,
        weights_history=pd.DataFrame(), portfolio_summary=portfolio_summary,
        call_allocation=call, beneficiary_allocation=beneficiary, regional_allocation=regional,
        scorecards=scorecards, diagnostics=diagnostics,
        interpretation=[
            "Policy-robust green projects remain selected from pure-score ranking through the full equity-and-cap model.",
            "Equity-sensitive gain/loss identifies projects whose decision changes when C1 and fairness rules are activated; it is not evidence of political merit or fault.",
            "The equity index is the share of selected funding directed to projects currently defined as disadvantaged.",
        ],
        settings={"policy_strength": policy_strength, "equity_floor": equity_floor, "call_budgets": dict(call_budgets), "beneficiary_category_caps": dict(beneficiary_category_caps or {})},
    )


def run_hybrid_ita(
    projects: pd.DataFrame,
    *,
    criterion_weights: Sequence[float],
    call_budgets: Mapping[str, float],
    beneficiary_category_caps: Mapping[str, float] | None = None,
    rounds: int = 4,
    simulations: int = 100,
    score_uncertainty: float = 1.5,
    final_gray_budget_factor: float = 0.85,
    green_threshold: float = 0.95,
    red_threshold: float = 0.05,
    equity_floor: float = 0.0,
    seed: int = 42,
    empirical_weight_vectors: Sequence[Sequence[float]] | None = None,
) -> ITAOutput:
    """Run simultaneous score and converging-weight uncertainty with freezing."""
    if rounds < 2 or simulations < 10:
        raise ValueError("Hybrid ITA requires at least two rounds and ten simulations per round.")
    if not (0 <= red_threshold < green_threshold <= 1):
        raise ValueError("Require 0 <= red threshold < green threshold <= 1.")
    if score_uncertainty < 0 or not 0 < final_gray_budget_factor <= 1:
        raise ValueError("Score uncertainty must be non-negative and the final budget factor must be in (0, 1].")
    weights = _normalise_weights(criterion_weights)
    empirical_weights: np.ndarray | None = None
    if empirical_weight_vectors is not None:
        empirical_weights = np.asarray(empirical_weight_vectors, dtype=float)
        if empirical_weights.ndim != 2 or empirical_weights.shape[1] != len(weights):
            raise ValueError("The empirical respondent matrix must have one column for every mapped ITA criterion.")
        if len(empirical_weights) < 4 or not np.isfinite(empirical_weights).all() or (empirical_weights < 0).any():
            raise ValueError("Empirical ITA weights require at least four complete, finite, non-negative respondent vectors.")
        row_sums = empirical_weights.sum(axis=1)
        if (row_sums <= 0).any():
            raise ValueError("Every empirical respondent weight vector must have a positive sum.")
        empirical_weights = empirical_weights / row_sums[:, None]
        weights = empirical_weights.mean(axis=0)
        weights = weights / weights.sum()
    criterion_columns = [f"C{i + 1}" for i in range(len(weights))]
    if not set(criterion_columns).issubset(projects.columns):
        raise ValueError("Prepared projects do not contain the criterion matrix expected by the weight vector.")
    matrix = projects[criterion_columns].to_numpy(float)
    caps = _beneficiary_cap_map(projects, beneficiary_category_caps)
    rng = np.random.default_rng(int(seed))
    n = len(projects)
    fixed_green: set[int] = set()
    fixed_red: set[int] = set(np.flatnonzero(~projects.eligible.to_numpy(bool)))
    decision_round = np.full(n, np.nan)
    decision = np.full(n, "Gray", dtype=object)
    locked_factor = np.full(n, np.nan)
    score_sensitive = np.zeros(n, dtype=bool)
    weight_sensitive = np.zeros(n, dtype=bool)
    history_rows: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    round_rows: list[dict] = []

    for round_number in range(1, rounds + 1):
        progress = (round_number - 1) / (rounds - 1)
        uncertainty = score_uncertainty * (1.0 - progress)
        gray_factor = 1.0 - (1.0 - final_gray_budget_factor) * progress
        if empirical_weights is None:
            weight_sets = converging_weight_sets(weights, round_number, rounds)
        else:
            scenario_count = min(int(simulations), len(empirical_weights))
            chosen = rng.choice(len(empirical_weights), size=scenario_count, replace=False)
            sampled_vectors = empirical_weights[chosen]
            weight_sets = (1.0 - progress) * sampled_vectors + progress * weights
            weight_sets = weight_sets / weight_sets.sum(axis=1, keepdims=True)
            weight_sets = np.unique(np.round(weight_sets, 12), axis=0)
        for k, vector in enumerate(weight_sets, start=1):
            for criterion, value in zip(criterion_columns, vector):
                weight_rows.append({"round": round_number, "weight_scenario": k, "criterion": criterion, "weight": float(value)})
        costs = projects.requested_budget.to_numpy(float) * np.where(np.isfinite(locked_factor), locked_factor, gray_factor)
        fixed_in = sorted(fixed_green); fixed_out = sorted(fixed_red)
        joint_draws = np.zeros((simulations, n), dtype=np.int8)
        score_draws = np.zeros((simulations, n), dtype=np.int8)
        weight_draws = np.zeros((len(weight_sets), n), dtype=np.int8)

        for scenario, vector in enumerate(weight_sets):
            utility = matrix @ vector
            weight_draws[scenario], _ = solve_portfolio(projects, utility, call_budgets=call_budgets, beneficiary_caps=caps, costs=costs, fixed_in=fixed_in, fixed_out=fixed_out, equity_floor=equity_floor)
        for simulation in range(simulations):
            sampled = np.clip(matrix + rng.uniform(-uncertainty, uncertainty, size=matrix.shape), 0.0, 10.0) if uncertainty > 0 else matrix
            score_utility = sampled @ weights
            score_draws[simulation], _ = solve_portfolio(projects, score_utility, call_budgets=call_budgets, beneficiary_caps=caps, costs=costs, fixed_in=fixed_in, fixed_out=fixed_out, equity_floor=equity_floor)
            vector = weight_sets[int(rng.integers(0, len(weight_sets)))]
            joint_draws[simulation], _ = solve_portfolio(projects, sampled @ vector, call_budgets=call_budgets, beneficiary_caps=caps, costs=costs, fixed_in=fixed_in, fixed_out=fixed_out, equity_floor=equity_floor)

        joint_probability = joint_draws.mean(axis=0)
        score_probability = score_draws.mean(axis=0)
        weight_probability = weight_draws.mean(axis=0)
        joint_state = _classify_probability(joint_probability, green_threshold, red_threshold)
        if round_number == rounds:
            joint_state = np.where(joint_probability >= 0.5, "Green", "Red")
        score_sensitive |= (score_probability > red_threshold) & (score_probability < green_threshold)
        weight_sensitive |= (weight_probability > red_threshold) & (weight_probability < green_threshold)
        unresolved = np.isnan(decision_round)
        newly_green = set(np.flatnonzero(unresolved & (joint_state == "Green")))
        newly_red = set(np.flatnonzero(unresolved & (joint_state == "Red")))
        for index in newly_green:
            decision_round[index] = round_number; decision[index] = "Green"; locked_factor[index] = gray_factor
        for index in newly_red:
            decision_round[index] = round_number; decision[index] = "Red"; locked_factor[index] = gray_factor
        fixed_green |= newly_green; fixed_red |= newly_red
        history_rows.append(pd.DataFrame({
            "project_id": projects.project_id, "round": round_number,
            "joint_inclusion_probability": joint_probability,
            "score_inclusion_probability": score_probability,
            "weight_inclusion_probability": weight_probability,
            "round_classification": joint_state,
        }))
        round_rows.append({
            "round": round_number, "score_uncertainty_half_width": uncertainty,
            "gray_budget_factor": gray_factor, "new_green": len(newly_green), "new_red": len(newly_red),
            "remaining_gray": int(np.isnan(decision_round).sum()), "simulations": simulations,
        })

    final_selected = (decision == "Green").astype(int)
    final_factor = np.where(np.isfinite(locked_factor), locked_factor, final_gray_budget_factor)
    result = projects.copy()
    result["hybrid_selected"] = final_selected
    result["decision_round"] = decision_round.astype(int)
    result["decision"] = decision
    result["assigned_budget_factor"] = final_factor
    result["allocated_budget"] = result.requested_budget * result.assigned_budget_factor * result.hybrid_selected
    result["score_sensitive"] = score_sensitive
    result["weight_sensitive"] = weight_sensitive
    result["uncertainty_zone"] = np.select(
        [
            (final_selected == 1) & ~score_sensitive & ~weight_sensitive,
            (final_selected == 0) & ~score_sensitive & ~weight_sensitive,
            score_sensitive & ~weight_sensitive,
            ~score_sensitive & weight_sensitive,
            score_sensitive & weight_sensitive,
        ],
        ["Stable green", "Stable red", "Score-sensitive", "Weight-sensitive", "Score-and-weight-sensitive"],
        default="Stable red",
    )
    history = pd.concat(history_rows, ignore_index=True)
    call, beneficiary, regional = _allocation_tables(result, "hybrid_selected")
    scorecards = _scorecard_table(result, "hybrid_selected", "uncertainty_zone")
    selected_rounds = result.loc[result.hybrid_selected.eq(1), "decision_round"]
    robustness = float(np.mean([(selected_rounds <= r).mean() for r in range(1, rounds)])) if len(selected_rounds) else np.nan
    allocated = float(result.allocated_budget.sum())
    disadvantaged_allocated = float(result.loc[result.disadvantaged, "allocated_budget"].sum())
    portfolio_summary = pd.DataFrame([{
        "selected_projects": int(final_selected.sum()), "allocated_budget": allocated,
        "unallocated_budget": sum(float(v) for v in call_budgets.values()) - allocated,
        "equity_index": disadvantaged_allocated / allocated if allocated else np.nan,
        "robustness_index": robustness, "rounds": rounds, "simulations_per_round": simulations,
    }])
    diagnostics = pd.DataFrame([
        {"check": "Stable green", "value": int((result.uncertainty_zone == "Stable green").sum())},
        {"check": "Score-sensitive", "value": int((result.uncertainty_zone == "Score-sensitive").sum())},
        {"check": "Weight-sensitive", "value": int((result.uncertainty_zone == "Weight-sensitive").sum())},
        {"check": "Sensitive to both", "value": int((result.uncertainty_zone == "Score-and-weight-sensitive").sum())},
        {"check": "Stable red", "value": int((result.uncertainty_zone == "Stable red").sum())},
    ])
    return ITAOutput(
        variant="Hybrid ITA-RW", projects=result, rounds=pd.DataFrame(round_rows),
        inclusion_history=history, weights_history=pd.DataFrame(weight_rows), portfolio_summary=portfolio_summary,
        call_allocation=call, beneficiary_allocation=beneficiary, regional_allocation=regional,
        scorecards=scorecards, diagnostics=diagnostics,
        interpretation=[
            "The joint inclusion probability is the proportion of score-and-weight scenarios in which each project enters an optimal feasible portfolio.",
            "Green and red decisions are frozen between rounds; only unresolved gray projects are reconsidered as uncertainty narrows and weights converge.",
            "The 2x2 uncertainty zone separates sensitivity to evaluator scores from sensitivity to policy weights; it does not measure implementation risk.",
            "The robustness index is the mean cumulative share of final green projects identified before the final round, consistent with the published ITA area interpretation.",
            ("Weight scenarios are sampled from the activated respondent-level empirical distribution and contract towards its empirical centre across rounds."
             if empirical_weights is not None else
             "Weight scenarios follow the published deterministic converging-weight construction around the mapped central weights."),
        ],
        settings={
            "criterion_weights": list(map(float, weights)), "call_budgets": dict(call_budgets),
            "beneficiary_category_caps": dict(beneficiary_category_caps or {}), "rounds": rounds,
            "simulations": simulations, "score_uncertainty": score_uncertainty,
            "final_gray_budget_factor": final_gray_budget_factor, "green_threshold": green_threshold,
            "red_threshold": red_threshold, "equity_floor": equity_floor, "seed": int(seed),
            "weight_scenario_source": "empirical_respondent_distribution" if empirical_weights is not None else "published_converging_schedule",
            "empirical_respondents": int(len(empirical_weights)) if empirical_weights is not None else 0,
        },
    )


def compare_portfolios(projects: pd.DataFrame, policy: ITAOutput, hybrid: ITAOutput) -> pd.DataFrame:
    rows: list[dict] = []
    candidates = [
        ("Conventional score optimisation", policy.projects.conventional_selected, policy.projects.requested_budget),
        ("ITA-PB", policy.projects.policy_selected, policy.projects.requested_budget),
        ("Hybrid ITA-RW", hybrid.projects.hybrid_selected, hybrid.projects.allocated_budget),
    ]
    if projects.actual_selected.notna().any():
        candidates.insert(0, ("Observed allocation", projects.actual_selected.fillna(0).astype(int), projects.requested_budget))
    for label, selected, costs in candidates:
        chosen = np.asarray(selected, dtype=int)
        expense = np.asarray(costs, dtype=float) * chosen if label != "Hybrid ITA-RW" else np.asarray(costs, dtype=float)
        total = float(expense.sum())
        equity = float(expense[projects.disadvantaged.to_numpy(bool)].sum()) / total if total else np.nan
        rows.append({"portfolio": label, "selected_projects": int(chosen.sum()), "allocated_budget": total, "mean_selected_score": float(projects.loc[chosen == 1, "final_score"].mean()) if chosen.sum() else np.nan, "equity_index": equity})
    return pd.DataFrame(rows)


def gams_model_text(output: ITAOutput) -> str:
    """Build an executable GAMS MIP representation of the final portfolio."""
    projects = output.projects.reset_index(drop=True)
    n = len(projects)
    calls = list(dict.fromkeys(projects.call.astype(str)))
    beneficiaries = list(dict.fromkeys(projects.beneficiary.astype(str)))
    call_keys = {name: f"c{i + 1}" for i, name in enumerate(calls)}
    beneficiary_keys = {name: f"b{i + 1}" for i, name in enumerate(beneficiaries)}
    call_members = ",".join(f"p{i + 1}.{call_keys[row.call]}" for i, row in projects.iterrows())
    beneficiary_members = ",".join(f"p{i + 1}.{beneficiary_keys[row.beneficiary]}" for i, row in projects.iterrows())
    values = ",\n".join(f"p{i + 1} {float(row.final_score):.10g}" for i, row in projects.iterrows())
    costs = ",\n".join(f"p{i + 1} {float(row.requested_budget):.10g}" for i, row in projects.iterrows())
    disadvantaged = ",".join(f"p{i + 1}" for i, row in projects.iterrows() if bool(row.disadvantaged)) or ""
    budgets = {normalise_call_code(k): float(v) for k, v in output.settings.get("call_budgets", {}).items()}
    call_caps = ",\n".join(f"{call_keys[name]} {budgets.get(normalise_call_code(name), 0):.10g}" for name in calls)
    category_caps = output.settings.get("beneficiary_category_caps", {})
    beneficiary_caps = _beneficiary_cap_map(projects, category_caps)
    beneficiary_cap_text = ",\n".join(f"{beneficiary_keys[name]} {beneficiary_caps.get(name, 1e30):.10g}" for name in beneficiaries)
    equity_floor = float(output.settings.get("equity_floor", 0.0))
    return f"""* Makryvelios ITA final-portfolio model. Generated reproducibly by the dashboard.
Sets
 i projects /p1*p{n}/
 c calls /c1*c{len(calls)}/
 b beneficiaries /b1*b{len(beneficiaries)}/
 ic(i,c) /{call_members}/
 ib(i,b) /{beneficiary_members}/
 disadvantaged(i) /{disadvantaged}/;

Parameters value(i) /
{values}/
/, cost(i) /
{costs}/
/, callCap(c) /
{call_caps}/
/, beneficiaryCap(b) /
{beneficiary_cap_text}/;
Scalar equityFloor /{equity_floor:.10g}/;

Binary Variable x(i);
Variable z;
Equations objective, callLimit(c), beneficiaryLimit(b), equityRequirement;
objective.. z =e= sum(i, value(i)*x(i));
callLimit(c).. sum(i$ic(i,c), cost(i)*x(i)) =l= callCap(c);
beneficiaryLimit(b).. sum(i$ib(i,b), cost(i)*x(i)) =l= beneficiaryCap(b);
equityRequirement$(equityFloor > 0)..
  sum(i$disadvantaged(i), cost(i)*x(i)) =g= equityFloor*sum(i, cost(i)*x(i));
Model ITA /all/;
option optcr=0;
Solve ITA using MIP maximizing z;
Display z.l, x.l;
"""


def ita_export_bundle(output: ITAOutput) -> bytes:
    """Return complete tables, provenance, scorecards and a GAMS-ready model."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        tables = {
            "projects.csv": output.projects,
            "rounds.csv": output.rounds,
            "inclusion_history.csv": output.inclusion_history,
            "weights_history.csv": output.weights_history,
            "portfolio_summary.csv": output.portfolio_summary,
            "call_allocation.csv": output.call_allocation,
            "beneficiary_allocation.csv": output.beneficiary_allocation,
            "regional_allocation.csv": output.regional_allocation,
            "project_scorecards.csv": output.scorecards,
            "diagnostics.csv": output.diagnostics,
        }
        for name, table in tables.items():
            archive.writestr(name, table.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("ita_final_model.gms", gams_model_text(output).encode("utf-8"))
        archive.writestr("settings.json", json.dumps(output.settings, ensure_ascii=False, indent=2).encode("utf-8"))
        archive.writestr("interpretation.txt", "\n".join(output.interpretation).encode("utf-8"))
        archive.writestr("README.txt", (
            f"{output.variant} reproducibility package\n\n"
            "The CSV files preserve every input, round, inclusion probability, allocation and scorecard.\n"
            "ita_final_model.gms is a transparent MIP representation for independent GAMS/GUROBI replication.\n"
            "The dashboard executes the equivalent binary model with SciPy/HiGHS; it does not claim that a GAMS licence is present.\n"
        ).encode("utf-8"))
    return buffer.getvalue()
