"""Portable analysis recipes for Makryvelios v5.9 confirmatory methods.

A recipe is deliberately plain JSON: no paper name, no dataset-specific code and
no executable Python.  It stores only a method key plus explicit variable
mapping/options.  The same recipe can therefore be rerun on a compatible future
dataset without rebuilding the application.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from confirmatory_analytics import (
    alexander_govern_test, beta_regression, brant_type_wald,
    compositional_transforms, conditional_logistic, cox_proportional_hazards,
    dirichlet_regression, dirichlet_component_alpha_regression, dunn_posthoc, equivalence_tost, exact_2x2_tests,
    firth_logistic, gee_regression, heckman_two_step, latent_class_analysis, latent_class_model_selection,
    linear_mixed_effects, mantel_haenszel, mca_ward, meta_analysis,
    multinomial_logit, ordered_regression, page_trend, permanova,
    plackett_luce, plackett_luce_mixture, plackett_luce_model_selection, rasch_1pl,
    regression_discontinuity, repeated_rank_tests, synthetic_control,
    tobit_regression, zero_inflated_count, brunner_munzel_test,
    jonckheere_terpstra, quade_test, cochran_q_test, mcnemar_test,
    bowker_symmetry_test, distance_correlation_test, energy_two_sample_test,
    partial_correlation, meta_regression, parsed_numeric_audit,
)

ENGINE_VERSION = "5.9.1"

RUNNERS = {
    "firth_logistic": firth_logistic,
    "ordered_regression": ordered_regression,
    "brant_type_wald": brant_type_wald,
    "multinomial_logit": multinomial_logit,
    "beta_regression": beta_regression,
    "tobit_regression": tobit_regression,
    "zero_inflated_count": zero_inflated_count,
    "linear_mixed_effects": linear_mixed_effects,
    "gee_regression": gee_regression,
    "cox_proportional_hazards": cox_proportional_hazards,
    "compositional_transforms": compositional_transforms,
    "permanova": permanova,
    "dirichlet_regression": dirichlet_regression,
    "dirichlet_component_alpha_regression": dirichlet_component_alpha_regression,
    "repeated_rank_tests": repeated_rank_tests,
    "plackett_luce": plackett_luce,
    "plackett_luce_mixture": plackett_luce_mixture,
    "plackett_luce_model_selection": plackett_luce_model_selection,
    "mca_ward": mca_ward,
    "latent_class_analysis": latent_class_analysis,
    "latent_class_model_selection": latent_class_model_selection,
    "dunn_posthoc": dunn_posthoc,
    "equivalence_tost": equivalence_tost,
    "meta_analysis": meta_analysis,
    "rasch_1pl": rasch_1pl,
    "regression_discontinuity": regression_discontinuity,
    "conditional_logistic": conditional_logistic,
    "exact_2x2_tests": exact_2x2_tests,
    "mantel_haenszel": mantel_haenszel,
    "page_trend": page_trend,
    "alexander_govern_test": alexander_govern_test,
    "heckman_two_step": heckman_two_step,
    "synthetic_control": synthetic_control,
    "brunner_munzel_test": brunner_munzel_test,
    "jonckheere_terpstra": jonckheere_terpstra,
    "quade_test": quade_test,
    "cochran_q_test": cochran_q_test,
    "mcnemar_test": mcnemar_test,
    "bowker_symmetry_test": bowker_symmetry_test,
    "distance_correlation_test": distance_correlation_test,
    "energy_two_sample_test": energy_two_sample_test,
    "partial_correlation": partial_correlation,
    "meta_regression": meta_regression,
    "parsed_numeric_audit": parsed_numeric_audit,
}

TEMPLATES: dict[str, dict[str, Any]] = {
    "firth_logistic": {"y": "binary_outcome", "x_vars": ["x1", "x2"], "categorical": []},
    "ordered_regression": {"y": "ordinal_outcome", "x_vars": ["x1"], "categorical": [], "distribution": "logit"},
    "brant_type_wald": {"y": "ordinal_outcome", "x_vars": ["x1"], "categorical": []},
    "multinomial_logit": {"y": "nominal_outcome", "x_vars": ["x1"], "categorical": []},
    "beta_regression": {"y": "fraction_outcome", "x_vars": ["x1"], "categorical": [], "precision_vars": []},
    "tobit_regression": {"y": "censored_outcome", "x_vars": ["x1"], "categorical": [], "lower": 0.0, "upper": None},
    "zero_inflated_count": {"y": "count_outcome", "x_vars": ["x1"], "inflation_vars": ["x1"], "categorical": [], "model": "ZIP"},
    "linear_mixed_effects": {"y": "outcome", "x_vars": ["x1"], "group": "subject_id", "categorical": [], "random_slope": None},
    "gee_regression": {"y": "outcome", "x_vars": ["x1"], "group": "subject_id", "categorical": [], "family": "Gaussian", "correlation": "Exchangeable"},
    "cox_proportional_hazards": {"time": "follow_up", "event": "event_01", "x_vars": ["x1"], "categorical": [], "strata": None},
    "compositional_transforms": {"columns": ["part1", "part2", "part3"], "zero_replacement": 1e-6},
    "permanova": {"columns": ["part1", "part2", "part3"], "group": "group", "transform": "ILR (Aitchison)", "permutations": 999, "seed": 42, "zero_replacement": 1e-6},
    "dirichlet_regression": {"components": ["part1", "part2", "part3"], "x_vars": ["x1"], "categorical": [], "reference_levels": {}, "zero_replacement": 1e-6},
    "dirichlet_component_alpha_regression": {"components": ["part1", "part2", "part3"], "x_vars": ["x1"], "categorical": [], "reference_levels": {}, "standardize_numeric": [], "zero_replacement": 1e-6, "likelihood_ratio_blocks": True},
    "repeated_rank_tests": {"columns": ["item1", "item2", "item3"], "higher_is_better": True, "adjustment": "holm"},
    "plackett_luce": {"rank_columns": ["rank_item1", "rank_item2", "rank_item3"]},
    "plackett_luce_mixture": {"rank_columns": ["rank_item1", "rank_item2", "rank_item3"], "components": 2, "seed": 42},
    "plackett_luce_model_selection": {"rank_columns": ["rank_item1", "rank_item2", "rank_item3"], "max_components": 5, "seed": 42, "n_init": 5, "criterion": "aic"},
    "mca_ward": {"categorical_columns": ["cat1", "cat2", "cat3"], "dimensions": 5, "clusters": 3, "ward_dimensions": 2, "benzecri": True},
    "latent_class_analysis": {"categorical_columns": ["cat1", "cat2", "cat3"], "classes": 3, "seed": 42, "n_init": 5},
    "latent_class_model_selection": {"categorical_columns": ["cat1", "cat2", "cat3"], "min_classes": 2, "max_classes": 5, "seed": 42, "n_init": 10, "criterion": "bic"},
    "dunn_posthoc": {"value": "outcome", "group": "group", "adjustment": "holm"},
    "equivalence_tost": {"value_a": "a", "value_b": "b", "group": None, "low": -0.2, "high": 0.2, "paired": True},
    "meta_analysis": {"effect": "effect", "standard_error": "se", "study_label": "study"},
    "rasch_1pl": {"item_columns": ["item1", "item2", "item3"]},
    "regression_discontinuity": {"y": "outcome", "running": "running", "cutoff": 0.0, "bandwidth": 1.0, "covariates": [], "kernel": "triangular"},
    "conditional_logistic": {"y": "outcome_01", "x_vars": ["x1"], "strata": "matched_set"},
    "exact_2x2_tests": {"outcome": "outcome_01", "exposure": "exposure_01"},
    "mantel_haenszel": {"outcome": "outcome_01", "exposure": "exposure_01", "strata": "stratum"},
    "page_trend": {"columns": ["ordered_condition1", "ordered_condition2", "ordered_condition3"], "ranked": False},
    "alexander_govern_test": {"value": "outcome", "group": "group"},
    "heckman_two_step": {"y": "observed_outcome", "selection": "selected_01", "outcome_predictors": ["x1"], "selection_predictors": ["x1", "exclusion_variable"], "categorical": []},
    "synthetic_control": {"unit": "unit", "time": "time", "outcome": "outcome", "treated_unit": "treated", "intervention_time": 2020},
    "brunner_munzel_test": {"value": "outcome", "group": "group", "alternative": "two-sided"},
    "jonckheere_terpstra": {"value": "outcome", "group": "ordered_group", "order": None, "alternative": "increasing", "permutations": 1999, "seed": 42},
    "quade_test": {"columns": ["condition1", "condition2", "condition3"]},
    "cochran_q_test": {"columns": ["binary1", "binary2", "binary3"]},
    "mcnemar_test": {"variable_a": "before_01", "variable_b": "after_01", "exact": True},
    "bowker_symmetry_test": {"variable_a": "before_category", "variable_b": "after_category"},
    "distance_correlation_test": {"x_columns": ["x1"], "y_columns": ["y1"], "permutations": 999, "seed": 42},
    "energy_two_sample_test": {"columns": ["x1", "x2"], "group": "group", "permutations": 999, "seed": 42},
    "partial_correlation": {"x": "x", "y": "y", "controls": ["z1"], "method": "pearson"},
    "meta_regression": {"effect": "effect", "standard_error": "se", "predictors": ["moderator"], "categorical": []},
    "parsed_numeric_audit": {"raw_column": "raw", "normalised_column": "normalised", "flag_column": "parsing_flag", "direct_flags": ["numeric_factor"]},
}


def make_recipe(method: str, kwargs: Mapping[str, Any] | None = None, *, name: str = "") -> dict[str, Any]:
    if method not in RUNNERS:
        raise ValueError(f"Unknown recipe method: {method}")
    return {
        "schema": "makryvelios.analysis_recipe.v1",
        "engine_version": ENGINE_VERSION,
        "name": name or method,
        "method": method,
        "kwargs": dict(kwargs if kwargs is not None else TEMPLATES.get(method, {})),
    }


def recipe_json(method: str, kwargs: Mapping[str, Any] | None = None, *, name: str = "") -> str:
    return json.dumps(make_recipe(method, kwargs, name=name), ensure_ascii=False, indent=2, default=str)


def parse_recipe(payload: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        recipe = dict(payload)
    else:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        recipe = json.loads(payload)
    if recipe.get("schema") != "makryvelios.analysis_recipe.v1":
        raise ValueError("Unsupported or missing Makryvelios recipe schema.")
    method = recipe.get("method")
    if method not in RUNNERS:
        raise ValueError(f"Recipe method {method!r} is not installed in this engine.")
    if not isinstance(recipe.get("kwargs"), dict):
        raise ValueError("Recipe kwargs must be a JSON object.")
    return recipe


def run_recipe(df: pd.DataFrame, payload: str | bytes | Mapping[str, Any]):
    recipe = parse_recipe(payload)
    return RUNNERS[recipe["method"]](df=df, **recipe["kwargs"])
