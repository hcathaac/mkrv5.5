"""Compatibility layer for the original 3,259-row Makryvelios R&D dataset."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from analytics_core import safe_name
from mapping import REGIONS, match_nuts2


def _index(df: pd.DataFrame) -> dict[str, str]:
    return {safe_name(c).casefold(): c for c in df.columns}


def find_col(df: pd.DataFrame, *aliases: str) -> str | None:
    idx = _index(df)
    for alias in aliases:
        key = safe_name(alias).casefold()
        if key in idx:
            return idx[key]
    for alias in aliases:
        key = safe_name(alias).casefold()
        candidates = [original for normal, original in idx.items() if key in normal or normal in key]
        if len(candidates) == 1:
            return candidates[0]
    return None


def is_rd_dataset(df: pd.DataFrame) -> bool:
    markers = [
        find_col(df, "A.A._Project"), find_col(df, "Project_start_year"),
        find_col(df, "Region"), find_col(df, "GDP_Region_End_Year"),
    ]
    return sum(x is not None for x in markers) >= 3


def rd_column_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    patterns = {
        "Project outcomes": ["Indicator_"],
        "Economic development": ["GDP_", "Cross_value_added", "Gross_value_added", "Employment_Region", "Number_of_business"],
        "R&D human capital": ["Employment_R_D", "Employment_R.D", "Researchers_", "Educational_institution", "Research_Center"],
        "R&D expenditure/intensity": ["R_D_exp", "Intestity_R_D", "InterstitY_R_D"],
        "Business innovation/collaboration": ["Percet_", "Rate_busin_", "Rate_turnover_", "Exp_innov_"],
        "Scientific output/networks": ["Numb_sc_pub", "Numb.sc", "Impact_index", "Numb_inter_collab"],
        "Project controls": ["R.T.D.I", "Scientific_Field", "Type_of_", "Role_of_", "Final_Project", "Final_Public", "Project_Duration", "absorption_rate"],
    }
    out = {}
    for label, tokens in patterns.items():
        out[label] = [c for c in df.columns if any(safe_name(t).casefold() in safe_name(c).casefold() for t in tokens)]
    return out


def build_region_year_panel(df: pd.DataFrame, allocation: str = "End-year") -> pd.DataFrame:
    region = find_col(df, "Region")
    start = find_col(df, "Project_start_year")
    end = find_col(df, "Project_end_year")
    project = find_col(df, "A.A._Project", "A_A_Project")
    if not all([region, start, end, project]):
        raise ValueError("The R&D region, project and year columns were not identified.")
    d = df.copy()
    d[start] = pd.to_numeric(d[start], errors="coerce")
    d[end] = pd.to_numeric(d[end], errors="coerce")
    if allocation.startswith("Active"):
        rows = []
        for _, row in d.dropna(subset=[start, end]).iterrows():
            years = list(range(int(row[start]), int(row[end]) + 1))
            for year in years:
                item = row.copy(); item["year"] = year; item["allocation_weight"] = 1 / max(len(years), 1)
                rows.append(item)
        d = pd.DataFrame(rows)
    else:
        d["year"] = d[start] if allocation.startswith("Start") else d[end]
        d["allocation_weight"] = 1.0
    d = d.dropna(subset=[region, "year"])

    budget = find_col(df, "Final Project's Budjet (at the end of the project)", "Final Projects Budget")
    public = find_col(df, "Final Public expenditure ( at the end)", "Final Public Expenditure")
    absorb_public = find_col(df, "(% absorption rate / public expenditure)")
    duration = find_col(df, "Project Duration  (year)")
    firm = find_col(df, "Indicator_3106_Nub_comp_benef")
    outcome_map = {
        "collaborative_projects": find_col(df, "Indicator_5_Nub_coop_comp_research_instit"),
        "patents": find_col(df, "Indicator_3115_Nub_of_patent"),
        "spin_offs": find_col(df, "Indicator_3111_Nub_spin_off_spin_outs"),
        "smes_benefited": find_col(df, "Indicator_3110_Num_of_SMES_benef"),
        "participating_firms": firm,
    }
    numeric_sources = [c for c in [budget, public, absorb_public, duration, *outcome_map.values()] if c]
    for c in numeric_sources:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if public and absorb_public:
        d["__absorbed_amount__"] = d[public] * d[absorb_public]
    if firm:
        d["__has_firm__"] = (d[firm].fillna(0) > 0).astype(float)
    if budget:
        d["__budget_alloc__"] = d[budget].fillna(0) * d["allocation_weight"]
    if public:
        d["__public_alloc__"] = d[public].fillna(0) * d["allocation_weight"]
    for output, c in outcome_map.items():
        if c:
            if output == "collaborative_projects":
                d[f"__{output}__"] = (d[c].fillna(0) > 0).astype(float) * d["allocation_weight"]
            else:
                d[f"__{output}__"] = d[c].fillna(0) * d["allocation_weight"]

    blocks = rd_column_blocks(df)
    region_covariates = list(dict.fromkeys(sum([blocks[k] for k in blocks if k not in {"Project outcomes", "Project controls"}], [])))
    region_covariates = [c for c in region_covariates if c in d and pd.to_numeric(d[c], errors="coerce").notna().sum() > 0]
    for c in region_covariates:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    aggregations: dict[str, str] = {project: "nunique"}
    for c in region_covariates:
        aggregations[c] = "mean"
    if "__absorbed_amount__" in d: aggregations["__absorbed_amount__"] = "sum"
    if "__budget_alloc__" in d: aggregations["__budget_alloc__"] = "sum"
    if "__public_alloc__" in d: aggregations["__public_alloc__"] = "sum"
    if duration: aggregations[duration] = "mean"
    if "__has_firm__" in d: aggregations["__has_firm__"] = "mean"
    for output in outcome_map:
        if f"__{output}__" in d: aggregations[f"__{output}__"] = "sum"
    panel = d.groupby([region, "year"], dropna=False).agg(aggregations).reset_index()
    panel = panel.rename(columns={region: "region", project: "project_count", duration: "average_project_duration", "__has_firm__": "share_projects_with_firms", "__budget_alloc__": "total_budget_allocated", "__public_alloc__": "total_public_expenditure_allocated"})
    for output in outcome_map:
        panel = panel.rename(columns={f"__{output}__": output})
    if "__absorbed_amount__" in panel and "total_public_expenditure_allocated" in panel:
        panel["regional_absorption_index"] = panel["__absorbed_amount__"] / panel["total_public_expenditure_allocated"].replace(0, np.nan)
        panel = panel.drop(columns="__absorbed_amount__")
    if "total_budget_allocated" in panel:
        panel["average_project_budget"] = panel["total_budget_allocated"] / panel["project_count"].replace(0, np.nan)
    panel["nuts_id"] = match_nuts2(panel["region"])
    panel = panel.merge(REGIONS[["nuts_id", "region_el", "region_en", "lat", "lon"]], on="nuts_id", how="left")
    return panel


def original_model_presets(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    end_covariates = [find_col(df, name) for name in [
        "GDP_Region_End_Year", "GDP_Region_per_person_End_Year", "Employment_Region_End_Year",
        "Researchers_numbers_Region_End_Year", "Researchers_number_Region_End_Year", "R_D_exp_int_Region_End_Year",
    ]]
    end_covariates = list(dict.fromkeys(c for c in end_covariates if c))
    rows = [
        ("EE1-project", find_col(df, "(% absorption rate / budget)"), "Project", "OLS HC3", end_covariates),
        ("EE2", find_col(df, "Indicator_6_Nub_of_resear_job_IPA"), "Project", "OLS HC3 / Poisson robustness", end_covariates),
        ("EE3", find_col(df, "Indicator_501_jobs_dur_the_operat_IPA"), "Project", "OLS HC3 / Poisson robustness", end_covariates),
        ("EE5", find_col(df, "Indicator_3106_Nub_comp_benef"), "Project", "Poisson/NB; OLS benchmark", end_covariates),
        ("EE1-panel", "regional_absorption_index", "Region–year", "Fractional logit / Poisson QMLE", []),
        ("EE4", "project_count", "Region–year", "Poisson FE / Negative binomial", []),
        ("EE6", "collaborative_projects", "Region–year", "Poisson FE / Negative binomial", []),
        ("EE7", "patents", "Region–year", "Poisson FE / Negative binomial", []),
        ("EE8", "spin_offs", "Region–year", "Poisson FE / Negative binomial", []),
        ("EE9", "smes_benefited", "Region–year", "Poisson FE / Negative binomial", []),
    ]
    return pd.DataFrame([{"model": model, "outcome": outcome, "unit": unit, "recommended_estimator": estimator, "core_regressors": "; ".join(xs)} for model, outcome, unit, estimator, xs in rows if outcome])
