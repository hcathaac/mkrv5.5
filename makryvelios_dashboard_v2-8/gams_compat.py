"""GAMS-compatible portfolio optimisation backend.

This module preserves the algebraic modelling logic used in Evangelos Makryvelios'
GAMS project-portfolio files while executing the equivalent binary MILP with the
open-source HiGHS solver exposed by SciPy.  GAMS remains a first-class export and
replication route; it is not required for routine execution.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


GAMS_STATUS_COLOURS = {
    "GREEN": "#16A34A",
    "RED": "#DC2626",
    "GRAY": "#6B7280",
    "GREY": "#6B7280",
    "UNCLASSIFIED": "#64748B",
    "SELECTED": "#2563EB",
}


SYN2_REGION_CAPS = {
    "EP2": 33_413_925.0,
    "ATT": 48_047_006.0,
    "CMK": 22_313_335.0,
    "WMK": 647_220.0,
    "STE": 3_480_311.0,
}
SYN2_SECTOR_CAPS = {
    "1": 11_664_951.0,
    "2": 11_664_951.0,
    "3": 11_664_951.0,
    "4": 8_748_713.0,
    "5": 11_664_951.0,
    "6": 8_748_713.0,
    "7": 8_748_713.0,
    "8": 11_644_951.0,
    "9": 8_748_713.0,
    "10": 14_581_189.0,
}
SYN2_WEIGHT_ROUNDS = {
    "Round 1": np.array([
        [1.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.000, 1.000],
    ]),
    "Round 2": np.array([
        [0.767, 0.100, 0.100],
        [0.100, 0.767, 0.100],
        [0.133, 0.133, 0.800],
    ]),
    "Round 3": np.array([
        [0.533, 0.200, 0.200],
        [0.200, 0.533, 0.200],
        [0.267, 0.267, 0.600],
    ]),
    "Round 4": np.array([
        [0.300, 0.300, 0.300],
        [0.300, 0.300, 0.300],
        [0.400, 0.400, 0.400],
    ]),
    "No ITA": np.array([
        [0.300, 0.300, 0.300],
        [0.300, 0.300, 0.300],
        [0.400, 0.400, 0.400],
    ]),
}

RND2437_REGIONS = ["ATT", "CMK", "EMK", "THE", "NAG", "EPI", "STE", "PEL", "CRE", "WGR", "WMK", "ION", "SAG"]
RND2437_REGION_GROUPS = {
    "lessdev": ["EMK", "CMK", "THE", "EPI", "WGR"],
    "trans": ["WMK", "CRE", "ION", "PEL", "NAG"],
}
RND2437_GROUP_CAPS = {"lessdev": 221_400_000.0, "trans": 61_500_000.0}
RND2437_REGION_CAPS = {"ATT": 106_600_000.0, "STE": 10_250_000.0, "SAG": 10_250_000.0}
RND2437_SECTOR_CAPS = {
    "1": 33_000_000.0, "2": 40_000_000.0, "3": 75_000_000.0, "4": 47_000_000.0,
    "5": 76_000_000.0, "6": 18_000_000.0, "7": 34_000_000.0, "8": 87_000_000.0,
}
RND2437_INTERVENTION_CAPS = {"1": 66_000_000.0, "2": 320_000_000.0, "3": 24_000_000.0}
RND2437_INTERVENTION_WEIGHTS = {
    "1": [0.2, 0.3, 0.5],
    "2": [0.4, 0.3, 0.3],
    "3": [0.3, 0.2, 0.5],
}
RND2437_MC_PRESETS = {
    "Round 1": {"iterations": 1000, "seed": 5780, "step": 1.0, "integer_low": -1, "integer_high": 1, "budget_factors": {}},
    "Round 2": {"iterations": 1000, "seed": 5780, "step": 0.5, "integer_low": -2, "integer_high": 2, "budget_factors": {}},
    "Round 3": {"iterations": 1000, "seed": 5780, "step": 0.5, "integer_low": -1, "integer_high": 1, "budget_factors": {"GREY": 0.925}},
    "Final": {"iterations": 1, "seed": 5780, "step": 0.0, "integer_low": -1, "integer_high": 1, "budget_factors": {"NEWGREEN": 0.925, "GREY2": 0.85}},
}


@dataclass
class GAMSCompatibleModel:
    projects: pd.DataFrame
    criteria: list[str]
    region_budget_columns: dict[str, str]
    region_caps: dict[str, float] = field(default_factory=dict)
    sector_column: str | None = None
    sector_caps: dict[str, float] = field(default_factory=dict)
    intervention_column: str | None = None
    intervention_caps: dict[str, float] = field(default_factory=dict)
    region_groups: dict[str, list[str]] = field(default_factory=dict)
    region_group_caps: dict[str, float] = field(default_factory=dict)
    status_column: str | None = None
    fixed_in: set[str] = field(default_factory=set)
    fixed_out: set[str] = field(default_factory=set)
    budget_factors: dict[str, float] = field(default_factory=dict)
    intervention_weights: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class GAMSRunResult:
    status: str
    objective: float
    project_results: pd.DataFrame
    region_allocation: pd.DataFrame
    sector_allocation: pd.DataFrame
    intervention_allocation: pd.DataFrame
    constraint_diagnostics: pd.DataFrame
    solver_message: str
    solver_status_code: int
    settings: dict


def preset_definition(name: str) -> dict:
    if name == "Vangelis – SYN2 540":
        return {
            "name": name,
            "criteria": ["C1", "C2", "C3"],
            "regions": list(SYN2_REGION_CAPS),
            "region_caps": SYN2_REGION_CAPS.copy(),
            "sector_caps": SYN2_SECTOR_CAPS.copy(),
            "weight_rounds": {k: v.copy() for k, v in SYN2_WEIGHT_ROUNDS.items()},
            "notes": [
                "Source files use 540 projects, 5 regional budget columns, 10 sectors and 3 criteria.",
                "Round 2 and Round 3 fix GREEN projects to 1 and RED projects to 0.",
                "Round 4 and the supplied No ITA source have the same weight matrix and active constraints; the application keeps them distinct but does not invent a difference.",
                "The first-round source uses CMK 22,312,335 while subsequent files use 22,313,335. The preset uses the later value and flags the source discrepancy.",
            ],
        }
    if name == "Vangelis – R&D 2437":
        return {
            "name": name,
            "criteria": ["C1", "C2", "C3"],
            "regions": RND2437_REGIONS.copy(),
            "region_caps": RND2437_REGION_CAPS.copy(),
            "region_groups": {k: v.copy() for k, v in RND2437_REGION_GROUPS.items()},
            "region_group_caps": RND2437_GROUP_CAPS.copy(),
            "sector_caps": RND2437_SECTOR_CAPS.copy(),
            "intervention_caps": RND2437_INTERVENTION_CAPS.copy(),
            "intervention_weights": {k: list(v) for k, v in RND2437_INTERVENTION_WEIGHTS.items()},
            "monte_carlo": {k: dict(v) for k, v in RND2437_MC_PRESETS.items()},
            "notes": [
                "Source files use 2,437 projects, 13 regions, 8 sectors, 3 interventions and 3 criteria.",
                "The objective uses intervention-specific criterion weights exactly as in the supplied GAMS files.",
                "Regional-group, sector and intervention budget constraints are retained as separate algebraic constraints.",
                "The supplied sequence uses seed 5780 and round-specific discrete score perturbations, with later rounds fixing GREEN/RED and reducing effective budgets for ambiguous/newly-green projects.",
            ],
        }
    return {"name": "Custom", "criteria": ["C1", "C2", "C3"], "regions": [], "region_caps": {}, "sector_caps": {}, "notes": []}


def _clean_id(value: object) -> str:
    return str(value).strip()


def _normalise_status(value: object) -> str:
    text = str(value).strip().upper()
    aliases = {"GREY": "GRAY", "G": "GREEN", "R": "RED", "Y": "GRAY"}
    return aliases.get(text, text or "UNCLASSIFIED")


def prepare_gams_compatible_model(
    data: pd.DataFrame,
    *,
    project_id_column: str,
    criterion_columns: Sequence[str],
    region_budget_columns: Mapping[str, str],
    region_caps: Mapping[str, float] | None = None,
    sector_column: str | None = None,
    sector_caps: Mapping[str, float] | None = None,
    intervention_column: str | None = None,
    intervention_caps: Mapping[str, float] | None = None,
    status_column: str | None = None,
    fixed_in: Sequence[str] = (),
    fixed_out: Sequence[str] = (),
    budget_factors: Mapping[str, float] | None = None,
    region_groups: Mapping[str, Sequence[str]] | None = None,
    region_group_caps: Mapping[str, float] | None = None,
    intervention_weights: Mapping[str, Sequence[float]] | None = None,
    metadata: Mapping | None = None,
) -> GAMSCompatibleModel:
    required = [project_id_column, *criterion_columns, *region_budget_columns.values()]
    if sector_column:
        required.append(sector_column)
    if intervention_column:
        required.append(intervention_column)
    if status_column:
        required.append(status_column)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing mapped columns: " + ", ".join(missing))
    if len(criterion_columns) < 1:
        raise ValueError("At least one criterion must be mapped.")
    frame = pd.DataFrame()
    frame["project_id"] = data[project_id_column].map(_clean_id)
    if frame.project_id.eq("").any():
        raise ValueError("Project IDs cannot be empty in the GAMS-compatible model.")
    duplicates = frame.project_id[frame.project_id.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError("Duplicate Project IDs detected. GAMS-style optimisation requires a unique project set. Examples: " + ", ".join(map(str, duplicates[:8])))
    criteria: list[str] = []
    for i, column in enumerate(criterion_columns, start=1):
        code = f"C{i}"
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Criterion {column!r} contains missing/non-numeric values. Correct or impute them before exact optimisation.")
        frame[code] = values.astype(float)
        criteria.append(code)
    mapped_regions: dict[str, str] = {}
    if len(set(map(str, region_budget_columns.values()))) != len(region_budget_columns):
        raise ValueError("Each GAMS region must map to a distinct source budget column.")
    for region, column in region_budget_columns.items():
        code = str(region).strip() or str(column)
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"Regional budget column {column!r} contains missing, non-numeric or negative values.")
        internal = f"budget__{code}"
        frame[internal] = values.astype(float)
        mapped_regions[code] = internal
    frame["total_budget"] = frame[list(mapped_regions.values())].sum(axis=1) if mapped_regions else 0.0
    if sector_column:
        frame["sector"] = data[sector_column].astype(str).str.strip()
    else:
        frame["sector"] = ""
    if intervention_column:
        frame["intervention"] = data[intervention_column].astype(str).str.strip()
    else:
        frame["intervention"] = ""
    if status_column:
        frame["ita_status"] = data[status_column].map(_normalise_status)
    else:
        frame["ita_status"] = "UNCLASSIFIED"
    fixed_in_set = {_clean_id(v) for v in fixed_in if _clean_id(v)}
    fixed_out_set = {_clean_id(v) for v in fixed_out if _clean_id(v)}
    overlap = fixed_in_set & fixed_out_set
    if overlap:
        raise ValueError("A project cannot be fixed both IN and OUT: " + ", ".join(sorted(overlap)[:8]))
    unknown_fixed = (fixed_in_set | fixed_out_set) - set(frame.project_id)
    if unknown_fixed:
        raise ValueError("Fixed project IDs not found in the mapped dataset: " + ", ".join(sorted(unknown_fixed)[:8]))
    factors = {_normalise_status(k): float(v) for k, v in (budget_factors or {}).items()}
    if any((not math.isfinite(v) or v <= 0) for v in factors.values()):
        raise ValueError("Budget factors must be finite and strictly positive.")
    iw = {str(k): list(map(float, v)) for k, v in (intervention_weights or {}).items()}
    for key, vec in iw.items():
        if len(vec) != len(criteria):
            raise ValueError(f"Intervention {key} has {len(vec)} weights but {len(criteria)} criteria are mapped.")
        arr = np.asarray(vec, float)
        if (arr < 0).any() or not np.isfinite(arr).all() or arr.sum() <= 0:
            raise ValueError(f"Intervention {key} weights must be finite, non-negative and sum to a positive value.")
        iw[key] = list(arr / arr.sum())
    return GAMSCompatibleModel(
        projects=frame.reset_index(drop=True),
        criteria=criteria,
        region_budget_columns=mapped_regions,
        region_caps={str(k): float(v) for k, v in (region_caps or {}).items() if float(v) >= 0},
        sector_column="sector" if sector_column else None,
        sector_caps={str(k): float(v) for k, v in (sector_caps or {}).items() if float(v) >= 0},
        intervention_column="intervention" if intervention_column else None,
        intervention_caps={str(k): float(v) for k, v in (intervention_caps or {}).items() if float(v) >= 0},
        region_groups={str(k): [str(x) for x in v] for k, v in (region_groups or {}).items()},
        region_group_caps={str(k): float(v) for k, v in (region_group_caps or {}).items() if float(v) >= 0},
        status_column="ita_status" if status_column else None,
        fixed_in=fixed_in_set,
        fixed_out=fixed_out_set,
        budget_factors=factors,
        intervention_weights=iw,
        metadata=dict(metadata or {}),
    )


def _effective_budget_matrix(model: GAMSCompatibleModel) -> pd.DataFrame:
    matrix = model.projects[list(model.region_budget_columns.values())].copy()
    if model.status_column and model.budget_factors:
        factors = model.projects[model.status_column].map(lambda s: model.budget_factors.get(str(s).upper(), 1.0)).to_numpy(float)
        matrix = matrix.mul(factors, axis=0)
    return matrix


def _utility_vector(model: GAMSCompatibleModel, weights: Sequence[float] | None = None, scores_override: np.ndarray | None = None) -> np.ndarray:
    scores = model.projects[model.criteria].to_numpy(float) if scores_override is None else np.asarray(scores_override, float)
    if scores.shape != (len(model.projects), len(model.criteria)):
        raise ValueError("Score matrix has incompatible dimensions.")
    if model.intervention_weights and model.intervention_column:
        utilities = np.zeros(len(model.projects), dtype=float)
        interventions = model.projects[model.intervention_column].astype(str).to_numpy()
        fallback = np.asarray(weights if weights is not None else np.ones(len(model.criteria)), float)
        fallback = fallback / fallback.sum()
        for i, code in enumerate(interventions):
            vec = np.asarray(model.intervention_weights.get(str(code), fallback), float)
            vec = vec / vec.sum()
            utilities[i] = float(scores[i] @ vec)
        return utilities
    vec = np.asarray(weights if weights is not None else np.ones(len(model.criteria)), dtype=float)
    if len(vec) != len(model.criteria) or (vec < 0).any() or not np.isfinite(vec).all() or vec.sum() <= 0:
        raise ValueError("Weight vector is incompatible with the mapped criteria.")
    vec = vec / vec.sum()
    return scores @ vec


def solve_gams_compatible(
    model: GAMSCompatibleModel,
    *,
    weights: Sequence[float] | None = None,
    scores_override: np.ndarray | None = None,
    mip_rel_gap: float = 0.0,
) -> GAMSRunResult:
    n = len(model.projects)
    if n == 0:
        raise ValueError("No projects available for optimisation.")
    utility = _utility_vector(model, weights, scores_override)
    budget_matrix = _effective_budget_matrix(model)
    total_budget = budget_matrix.sum(axis=1).to_numpy(float)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []

    for region, cap in model.region_caps.items():
        if region not in model.region_budget_columns:
            continue
        rows.append(budget_matrix[model.region_budget_columns[region]].to_numpy(float))
        lower.append(-np.inf); upper.append(float(cap)); names.append(f"region:{region}")
    for group, regions in model.region_groups.items():
        if group not in model.region_group_caps:
            continue
        cols = [model.region_budget_columns[r] for r in regions if r in model.region_budget_columns]
        if not cols:
            continue
        rows.append(budget_matrix[cols].sum(axis=1).to_numpy(float))
        lower.append(-np.inf); upper.append(float(model.region_group_caps[group])); names.append(f"region_group:{group}")
    if model.sector_column:
        sector_values = model.projects[model.sector_column].astype(str).to_numpy()
        for sector, cap in model.sector_caps.items():
            rows.append(np.where(sector_values == str(sector), total_budget, 0.0))
            lower.append(-np.inf); upper.append(float(cap)); names.append(f"sector:{sector}")
    if model.intervention_column:
        intv_values = model.projects[model.intervention_column].astype(str).to_numpy()
        for intv, cap in model.intervention_caps.items():
            rows.append(np.where(intv_values == str(intv), total_budget, 0.0))
            lower.append(-np.inf); upper.append(float(cap)); names.append(f"intervention:{intv}")

    lb = np.zeros(n, dtype=float)
    ub = np.ones(n, dtype=float)
    ids = model.projects.project_id.astype(str).to_numpy()
    for i, pid in enumerate(ids):
        status = str(model.projects.iloc[i].get("ita_status", "")).upper()
        if pid in model.fixed_in or status == "GREEN" and model.metadata.get("fix_green", False):
            lb[i] = ub[i] = 1.0
        if pid in model.fixed_out or status == "RED" and model.metadata.get("fix_red", False):
            lb[i] = ub[i] = 0.0

    constraints = None
    if rows:
        matrix = csr_matrix(np.vstack(rows))
        constraints = LinearConstraint(matrix, np.asarray(lower, float), np.asarray(upper, float))
    options = {"disp": False, "mip_rel_gap": max(0.0, float(mip_rel_gap))}
    result = milp(c=-utility, integrality=np.ones(n), bounds=Bounds(lb, ub), constraints=constraints, options=options)
    status_map = {0: "OPTIMAL", 1: "LIMIT_REACHED", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "ERROR"}
    status = status_map.get(int(result.status), "ERROR")
    selected = np.zeros(n, dtype=int)
    if result.x is not None:
        selected = (np.asarray(result.x) >= 0.5).astype(int)
    objective = float(utility @ selected)
    out = model.projects.copy()
    out["weighted_score"] = utility
    out["selected"] = selected
    out["decision"] = np.where(selected == 1, "SELECTED", "NOT SELECTED")
    out["effective_budget"] = total_budget
    out["allocated_budget"] = total_budget * selected

    region_rows = []
    for region, column in model.region_budget_columns.items():
        allocated = float((budget_matrix[column].to_numpy(float) * selected).sum())
        cap = model.region_caps.get(region, np.nan)
        region_rows.append({"region": region, "allocated_budget": allocated, "cap": cap, "remaining": float(cap - allocated) if pd.notna(cap) else np.nan, "utilisation": allocated / cap if pd.notna(cap) and cap else np.nan})
    region_allocation = pd.DataFrame(region_rows)

    sector_allocation = pd.DataFrame()
    if model.sector_column:
        temp = pd.DataFrame({"sector": model.projects[model.sector_column].astype(str), "allocated_budget": total_budget * selected, "selected": selected})
        sector_allocation = temp.groupby("sector", as_index=False).agg(allocated_budget=("allocated_budget", "sum"), selected_projects=("selected", "sum"))
        sector_allocation["cap"] = sector_allocation.sector.map(model.sector_caps)
        sector_allocation["remaining"] = sector_allocation.cap - sector_allocation.allocated_budget
        sector_allocation["utilisation"] = sector_allocation.allocated_budget / sector_allocation.cap

    intervention_allocation = pd.DataFrame()
    if model.intervention_column:
        temp = pd.DataFrame({"intervention": model.projects[model.intervention_column].astype(str), "allocated_budget": total_budget * selected, "selected": selected})
        intervention_allocation = temp.groupby("intervention", as_index=False).agg(allocated_budget=("allocated_budget", "sum"), selected_projects=("selected", "sum"))
        intervention_allocation["cap"] = intervention_allocation.intervention.map(model.intervention_caps)
        intervention_allocation["remaining"] = intervention_allocation.cap - intervention_allocation.allocated_budget
        intervention_allocation["utilisation"] = intervention_allocation.allocated_budget / intervention_allocation.cap

    diagnostics = []
    if rows:
        for name, row, cap in zip(names, rows, upper):
            used = float(np.asarray(row) @ selected)
            diagnostics.append({"constraint": name, "used": used, "cap": float(cap), "slack": float(cap - used), "utilisation": used / cap if cap else np.nan, "binding": bool(abs(cap - used) <= max(1.0, abs(cap) * 1e-7))})
    diag = pd.DataFrame(diagnostics)
    settings = {
        "solver": "SciPy/HiGHS MILP",
        "gams_compatible": True,
        "criteria": list(model.criteria),
        "weights": None if weights is None else list(map(float, weights)),
        "mip_rel_gap": float(mip_rel_gap),
        "fixed_in": sorted(model.fixed_in),
        "fixed_out": sorted(model.fixed_out),
        "budget_factors": dict(model.budget_factors),
        "region_caps": dict(model.region_caps),
        "sector_caps": dict(model.sector_caps),
        "intervention_caps": dict(model.intervention_caps),
        "region_groups": dict(model.region_groups),
        "region_group_caps": dict(model.region_group_caps),
        "intervention_weights": dict(model.intervention_weights),
        **model.metadata,
    }
    return GAMSRunResult(
        status=status, objective=objective, project_results=out,
        region_allocation=region_allocation, sector_allocation=sector_allocation,
        intervention_allocation=intervention_allocation, constraint_diagnostics=diag,
        solver_message=str(result.message), solver_status_code=int(result.status), settings=settings,
    )


def solve_weight_matrix(model: GAMSCompatibleModel, weight_matrix: pd.DataFrame, *, mip_rel_gap: float = 0.0) -> tuple[pd.DataFrame, dict[str, GAMSRunResult]]:
    if list(weight_matrix.index) != model.criteria:
        raise ValueError("Weight-matrix rows must be exactly " + ", ".join(model.criteria))
    results: dict[str, GAMSRunResult] = {}
    selections = pd.DataFrame({"project_id": model.projects.project_id})
    summary_rows = []
    for column in weight_matrix.columns:
        vec = weight_matrix[column].to_numpy(float)
        run = solve_gams_compatible(model, weights=vec, mip_rel_gap=mip_rel_gap)
        results[str(column)] = run
        selections[str(column)] = run.project_results.selected.to_numpy(int)
        summary_rows.append({"scenario": str(column), "solver_status": run.status, "portfolio_score": run.objective, "selected_projects": int(run.project_results.selected.sum()), "allocated_budget": float(run.project_results.allocated_budget.sum())})
    return pd.DataFrame(summary_rows), {**results, "__selection_matrix__": selections}


def monte_carlo_gams_compatible(
    model: GAMSCompatibleModel,
    *,
    weights: Sequence[float] | None = None,
    iterations: int = 1000,
    seed: int = 5780,
    perturbation_step: float = 0.5,
    integer_low: int = -2,
    integer_high: int = 2,
    score_min: float = 0.0,
    score_max: float = 5.0,
    green_threshold: float = 0.99,
    red_threshold: float = 0.01,
    mip_rel_gap: float = 0.0005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iterations = int(iterations)
    if iterations < 1 or iterations > 5000:
        raise ValueError("Monte Carlo iterations must be between 1 and 5,000.")
    rng = np.random.default_rng(int(seed))
    base_scores = model.projects[model.criteria].to_numpy(float)
    counts = np.zeros(len(model.projects), dtype=int)
    objective_values = np.zeros(iterations, dtype=float)
    selected_counts = np.zeros(iterations, dtype=int)
    for i in range(iterations):
        if perturbation_step == 0:
            perturbed = base_scores.copy()
        else:
            shocks = rng.integers(int(integer_low), int(integer_high) + 1, size=base_scores.shape)
            perturbed = np.clip(base_scores + float(perturbation_step) * shocks, float(score_min), float(score_max))
        run = solve_gams_compatible(model, weights=weights, scores_override=perturbed, mip_rel_gap=mip_rel_gap)
        if run.status not in {"OPTIMAL", "LIMIT_REACHED"}:
            raise RuntimeError(f"Monte Carlo iteration {i + 1} failed with solver status {run.status}: {run.solver_message}")
        selected = run.project_results.selected.to_numpy(int)
        counts += selected
        objective_values[i] = run.objective
        selected_counts[i] = int(selected.sum())
    freq = counts / iterations
    classification = np.where(freq >= green_threshold, "GREEN", np.where(freq <= red_threshold, "RED", "GRAY"))
    projects = pd.DataFrame({
        "project_id": model.projects.project_id,
        "selection_count": counts,
        "selection_frequency": freq,
        "ita_classification": classification,
    })
    draws = pd.DataFrame({"iteration": np.arange(1, iterations + 1), "portfolio_score": objective_values, "selected_projects": selected_counts})
    return projects, draws


def _gams_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", str(text))
    if not value or value[0].isdigit():
        value = "n_" + value
    return value[:55]


def _project_aliases(model: GAMSCompatibleModel) -> dict[str, str]:
    return {pid: f"p{i + 1}" for i, pid in enumerate(model.projects.project_id.astype(str))}


def gams_model_text(model: GAMSCompatibleModel, *, weights: Sequence[float] | None = None, monte_carlo: Mapping | None = None) -> str:
    aliases = _project_aliases(model)
    regions = list(model.region_budget_columns)
    sectors = sorted(model.projects.sector.astype(str).unique()) if model.sector_column else []
    interventions = sorted(model.projects.intervention.astype(str).unique()) if model.intervention_column else []
    ncrit = len(model.criteria)
    lines = [
        "* Makryvelios GAMS-compatible portfolio model generated by the application.",
        "* Canonical algebraic logic retained; routine execution may use SciPy/HiGHS.",
        "sets",
        f"    p projects /p1*p{len(model.projects)}/",
        f"    rg regions /{', '.join(_gams_name(r) for r in regions)}/" if regions else "    rg regions /dummy/",
        f"    crit criteria /1*{ncrit}/",
    ]
    if sectors:
        lines.append(f"    sec sectors /1*{len(sectors)}/")
    if interventions:
        lines.append(f"    intv intervention /1*{len(interventions)}/")
    for group, regs in model.region_groups.items():
        available = [r for r in regs if r in regions]
        if available:
            lines.append(f"    {_gams_name(group)}(rg) /{', '.join(_gams_name(r) for r in available)}/")
    green_aliases = [aliases[pid] for pid in model.projects.loc[model.projects.ita_status.eq("GREEN"), "project_id"]] if "ita_status" in model.projects else []
    red_aliases = [aliases[pid] for pid in model.projects.loc[model.projects.ita_status.eq("RED"), "project_id"]] if "ita_status" in model.projects else []
    gray_aliases = [aliases[pid] for pid in model.projects.loc[model.projects.ita_status.isin(["GRAY", "GREY"]), "project_id"]] if "ita_status" in model.projects else []
    lines.append(f"    GREEN(p) /{','.join(green_aliases)}/" if green_aliases else "    GREEN(p) //")
    lines.append(f"    RED(p) /{','.join(red_aliases)}/" if red_aliases else "    RED(p) //")
    lines.append(f"    GREY(p) /{','.join(gray_aliases)}/;" if gray_aliases else "    GREY(p) //;")
    lines.extend([
        "",
        "table budget(p,rg)",
        "$include \"budget.prn\";",
        "",
        "table score(p,crit)",
        "$include \"score.prn\";",
    ])
    if sectors:
        lines.extend(["", "parameter sector(p) /", "$include \"sector.prn\"", "/;"])
    if interventions:
        lines.extend(["", "parameter intervention(p) /", "$include \"intervention.prn\"", "/;"])
    lines.append("")
    if model.sector_caps:
        lines.append("parameter totbudg(sec) /")
        sector_ord = {s: i + 1 for i, s in enumerate(sectors)}
        for s, cap in model.sector_caps.items():
            if s in sector_ord:
                lines.append(f"{sector_ord[s]} {cap:.12g}")
        lines.append("/;")
    if model.intervention_caps:
        lines.append("parameter totbudgi(intv) /")
        intv_ord = {s: i + 1 for i, s in enumerate(interventions)}
        for s, cap in model.intervention_caps.items():
            if s in intv_ord:
                lines.append(f"{intv_ord[s]} {cap:.12g}")
        lines.append("/;")
    lines.extend([
        "parameter totscore(p);",
        "Binary Variables X(p);",
        "Variables PORTFSCORE;",
        "Equations",
    ])
    eq_names = []
    for region in model.region_caps:
        if region in model.region_budget_columns:
            eq_names.append(f"budget_{_gams_name(region)}")
    for group in model.region_group_caps:
        if group in model.region_groups:
            eq_names.append(f"budget_{_gams_name(group)}")
    if model.sector_caps and sectors:
        eq_names.append("budget_sec(sec)")
    if model.intervention_caps and interventions:
        eq_names.append("budget_intv(intv)")
    eq_names.append("totscore_eq")
    lines.extend([f"    {name}" for name in eq_names])
    lines.append(";")
    for region, cap in model.region_caps.items():
        if region in model.region_budget_columns:
            lines.append(f"budget_{_gams_name(region)}.. sum(p, budget(p,'{_gams_name(region)}')*X(p)) =l= {cap:.12g};")
    for group, cap in model.region_group_caps.items():
        if group in model.region_groups:
            lines.append(f"budget_{_gams_name(group)}.. sum((p,{_gams_name(group)}), budget(p,{_gams_name(group)})*X(p)) =l= {cap:.12g};")
    if model.sector_caps and sectors:
        lines.append("budget_sec(sec).. sum(p$(sector(p) eq ord(sec)), sum(rg,budget(p,rg))*X(p)) =l= totbudg(sec);")
    if model.intervention_caps and interventions:
        lines.append("budget_intv(intv).. sum(p$(intervention(p) eq ord(intv)), sum(rg,budget(p,rg))*X(p)) =l= totbudgi(intv);")
    lines.append("totscore_eq.. sum(p, X(p)*totscore(p)) =e= PORTFSCORE;")
    lines.append("model itanew /all/;")
    if model.intervention_weights and interventions:
        for intervention, vec in model.intervention_weights.items():
            terms = "+".join(f"{float(w):.10g}*score(p,'{i + 1}')" for i, w in enumerate(vec))
            lines.append(f"loop(p$(intervention(p)={intervention}), totscore(p)={terms});")
    else:
        vec = np.asarray(weights if weights is not None else np.ones(ncrit), float)
        vec = vec / vec.sum()
        terms = "+".join(f"{float(w):.10g}*score(p,'{i + 1}')" for i, w in enumerate(vec))
        lines.append(f"loop(p, totscore(p)={terms});")
    if model.metadata.get("fix_green", False):
        lines.append("X.fx(GREEN)=1;")
    if model.metadata.get("fix_red", False):
        lines.append("X.fx(RED)=0;")
    for status, factor in model.budget_factors.items():
        setname = "GREY" if status in {"GRAY", "GREY"} else _gams_name(status)
        lines.append(f"* Effective-budget rule preserved from configured status: {status}")
        if setname == "GREY":
            lines.append(f"loop(GREY, budget(GREY,rg)={float(factor):.10g}*budget(GREY,rg));")
    for pid in sorted(model.fixed_in):
        lines.append(f"X.fx('{aliases[pid]}')=1;")
    for pid in sorted(model.fixed_out):
        lines.append(f"X.fx('{aliases[pid]}')=0;")
    lines.append(f"option optcr={float(model.metadata.get('mip_rel_gap', 0.0)):.10g};")
    if monte_carlo:
        lines.extend([
            f"option seed={int(monte_carlo.get('seed', 5780))};",
            f"scalar MCiter /{int(monte_carlo.get('iterations', 1000))}/, iter, z1, z2, z3;",
            "* Monte Carlo execution retained conceptually. The dashboard also exports exact draw tables.",
        ])
    lines.extend([
        "Solve itanew using MIP maximizing PORTFSCORE;",
        "Display PORTFSCORE.l, X.l;",
        "",
    ])
    return "\n".join(lines)


def gams_data_files(model: GAMSCompatibleModel) -> dict[str, bytes]:
    aliases = _project_aliases(model)
    region_headers = list(model.region_budget_columns)
    gams_region_headers = [_gams_name(r) for r in region_headers]
    budget = _effective_budget_matrix(model)
    budget.columns = gams_region_headers
    budget.insert(0, "project", [aliases[p] for p in model.projects.project_id])
    budget_lines = ["             " + " ".join(f"{r:>12}" for r in gams_region_headers)]
    for _, row in budget.iterrows():
        budget_lines.append(f"{row['project']:>10} " + " ".join(f"{float(row[r]):12.6g}" for r in gams_region_headers))
    scores = model.projects[model.criteria]
    score_lines = ["             " + " ".join(f"{i + 1:>10}" for i in range(len(model.criteria)))]
    for idx, row in scores.iterrows():
        score_lines.append(f"{aliases[model.projects.iloc[idx].project_id]:>10} " + " ".join(f"{float(v):10.6g}" for v in row))
    files: dict[str, bytes] = {
        "budget.prn": ("\n".join(budget_lines) + "\n").encode("utf-8"),
        "score.prn": ("\n".join(score_lines) + "\n").encode("utf-8"),
    }
    if model.sector_column:
        sectors = sorted(model.projects.sector.astype(str).unique())
        ords = {s: i + 1 for i, s in enumerate(sectors)}
        text = "\n".join(f"{aliases[row.project_id]} {ords[str(row.sector)]}" for _, row in model.projects.iterrows()) + "\n"
        files["sector.prn"] = text.encode("utf-8")
    if model.intervention_column:
        intvs = sorted(model.projects.intervention.astype(str).unique())
        ords = {s: i + 1 for i, s in enumerate(intvs)}
        text = "\n".join(f"{aliases[row.project_id]} {ords[str(row.intervention)]}" for _, row in model.projects.iterrows()) + "\n"
        files["intervention.prn"] = text.encode("utf-8")
    for status, filename in [("GREEN", "green.txt"), ("RED", "red.txt"), ("GRAY", "gray.txt")]:
        ids = model.projects.loc[model.projects.ita_status.eq(status), "project_id"] if "ita_status" in model.projects else []
        files[filename] = ("\n".join(aliases[str(pid)] for pid in ids) + ("\n" if len(ids) else "")).encode("utf-8")
    crosswalk = pd.DataFrame({"gams_project": [aliases[p] for p in model.projects.project_id], "project_id": model.projects.project_id})
    files["project_crosswalk.csv"] = crosswalk.to_csv(index=False).encode("utf-8-sig")
    return files


def gams_reproducibility_bundle(
    model: GAMSCompatibleModel,
    *,
    weights: Sequence[float] | None = None,
    run: GAMSRunResult | None = None,
    monte_carlo_projects: pd.DataFrame | None = None,
    monte_carlo_draws: pd.DataFrame | None = None,
    monte_carlo_settings: Mapping | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.gms", gams_model_text(model, weights=weights, monte_carlo=monte_carlo_settings).encode("utf-8"))
        for name, payload in gams_data_files(model).items():
            archive.writestr(name, payload)
        settings = {
            "criteria": model.criteria,
            "region_caps": model.region_caps,
            "sector_caps": model.sector_caps,
            "intervention_caps": model.intervention_caps,
            "region_groups": model.region_groups,
            "region_group_caps": model.region_group_caps,
            "budget_factors": model.budget_factors,
            "fixed_in": sorted(model.fixed_in),
            "fixed_out": sorted(model.fixed_out),
            "intervention_weights": model.intervention_weights,
            "weights": None if weights is None else list(map(float, weights)),
            "metadata": model.metadata,
            "monte_carlo": dict(monte_carlo_settings or {}),
        }
        archive.writestr("settings.json", json.dumps(settings, indent=2, ensure_ascii=False).encode("utf-8"))
        if run is not None:
            archive.writestr("project_results.csv", run.project_results.to_csv(index=False).encode("utf-8-sig"))
            archive.writestr("region_allocation.csv", run.region_allocation.to_csv(index=False).encode("utf-8-sig"))
            archive.writestr("sector_allocation.csv", run.sector_allocation.to_csv(index=False).encode("utf-8-sig"))
            archive.writestr("intervention_allocation.csv", run.intervention_allocation.to_csv(index=False).encode("utf-8-sig"))
            archive.writestr("constraint_diagnostics.csv", run.constraint_diagnostics.to_csv(index=False).encode("utf-8-sig"))
        if monte_carlo_projects is not None:
            archive.writestr("monte_carlo_project_frequency.csv", monte_carlo_projects.to_csv(index=False).encode("utf-8-sig"))
        if monte_carlo_draws is not None:
            archive.writestr("monte_carlo_draws.csv", monte_carlo_draws.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("README.txt", (
            "Makryvelios GAMS-compatible portfolio package\n\n"
            "The model.gms file retains a GAMS-style algebraic representation and can be adapted for native GAMS execution.\n"
            "The live application executes the equivalent binary mixed-integer model with SciPy/HiGHS, avoiding a mandatory commercial GAMS licence.\n"
            "Project IDs are preserved in project_crosswalk.csv; raw source data are never silently reordered by row position.\n"
        ).encode("utf-8"))
    return buffer.getvalue()
