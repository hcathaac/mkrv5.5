from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics_core import (
    correlation_matrix, descriptive_statistics, fit_detailed_model,
    matrix_ols_many_outcomes, promote_embedded_header, quality_summary,
    tidy_frame, to_excel_bytes, regularised_regression,
    instrumental_variables_2sls, difference_in_differences, cronbach_alpha,
    monte_carlo_ols, monte_carlo_portfolio,
    outlier_summary,
)
from legacy_rd import build_region_year_panel, is_rd_dataset
from mapping import match_nuts2, moran_diagnostics, REGIONS
from visuals import (
    ols_publication_bundle, monte_carlo_publication_bundle,
    clustering_publication_bundle, predictive_publication_bundle,
    panel_publication_bundle,
)
from advanced_analytics import advanced_clustering, predictive_model_comparison, panel_model_suite
from mcda import ahp_weights, mcda_analysis, mcda_publication_bundle
from research_chair import (
    add_safe_derived_column, apply_scope, build_offline_reply,
    build_paper_blueprint, execute_protocol, execute_natural_language_command, research_bundle,
    select_pdf_evidence,
)
from output_guidance import build_output_guide
from prompt_library import prompt_library


def synthetic(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    group = rng.choice(["A", "B", "C"], size=n)
    y = 1 + 2 * x1 - .5 * x2 + rng.normal(scale=.4, size=n)
    count = rng.poisson(np.exp(.2 + .25 * x1))
    return pd.DataFrame({"y": y, "y2": y * .5 + rng.normal(size=n), "x1": x1, "x2": x2, "count": count, "group": group})


def test_embedded_header_promotion():
    raw = pd.DataFrame([["Project", "Region", "Budget"], [1, "Attica", 10], [2, "Crete", 20]], columns=[1, 2, 3])
    out = tidy_frame(raw, normalise_columns=True)
    assert list(out.columns) == ["Project", "Region", "Budget"]
    assert len(out) == 2
    assert pd.api.types.is_numeric_dtype(out["Budget"])


def test_descriptives_and_correlations():
    df = synthetic()
    desc = descriptive_statistics(df, ["y", "x1"])
    corr, p = correlation_matrix(df, ["y", "x1", "x2"])
    assert set(desc.variable) == {"y", "x1"}
    assert corr.loc["y", "x1"] > .8
    assert p.loc["y", "x1"] < .001
    outliers = outlier_summary(df, ["y", "x1"])
    assert set(outliers.variable) == {"y", "x1"}


def test_detailed_ols():
    df = synthetic()
    out = fit_detailed_model(df, "y", ["x1", "x2"], estimator="OLS", covariance="HC3")
    b = out.coefficients.set_index("term").coefficient
    assert abs(b["x1"] - 2) < .15
    assert abs(b["x2"] + .5) < .15
    assert int(out.fit.iloc[0].n) == len(df)
    assert not out.diagnostics.empty


def test_many_outcome_engine():
    df = synthetic()
    coef, fit = matrix_ols_many_outcomes(df, ["y", "y2"], ["x1", "x2"])
    assert len(coef) == 2 * 3
    assert len(fit) == 2
    assert fit.loc[fit.outcome == "y", "r_squared"].iloc[0] > .8


def test_excel_export():
    payload = to_excel_bytes({"Summary": quality_summary(synthetic())})
    assert payload[:2] == b"PK"
    book = pd.ExcelFile(io.BytesIO(payload))
    assert "Summary" in book.sheet_names
    assert "Output guide" in book.sheet_names


def test_greek_aliases_and_moran():
    aliases = pd.Series(["Attica", "Κρήτη", "Central Macedonia"])
    assert list(match_nuts2(aliases)) == ["EL30", "EL43", "EL52"]
    data = REGIONS.copy()
    data["metric"] = np.arange(len(data), dtype=float)
    global_table, local = moran_diagnostics(data, "metric", permutations=49)
    assert len(global_table) == 1
    assert len(local) == 13


def test_rd_panel_compatibility():
    df = pd.DataFrame({
        "A.A._Project": [1, 2, 3], "Project_start_year": [2020, 2020, 2021], "Project_end_year": [2021, 2021, 2022],
        "Project Duration  (year)": [1, 1, 1], "Region": ["Attica", "Attica", "Crete"],
        "Final Project's Budjet (at the end of the project)": [100, 200, 300],
        "Final Public expenditure ( at the end)": [80, 150, 250], "(% absorption rate / public expenditure)": [.8, .9, .7],
        "GDP_Region_End_Year": [10, 10, 8], "Indicator_5_Nub_coop_comp_research_instit": [1, 0, 2],
        "Indicator_3106_Nub_comp_benef": [2, 0, 1], "Indicator_3115_Nub_of_patent": [1, 0, 0],
        "Indicator_3111_Nub_spin_off_spin_outs": [0, 0, 1], "Indicator_3110_Num_of_SMES_benef": [2, 1, 3],
    })
    assert is_rd_dataset(df)
    panel = build_region_year_panel(df, "End-year")
    attica = panel[(panel.region == "Attica") & (panel.year == 2021)].iloc[0]
    assert attica.project_count == 2
    assert attica.collaborative_projects == 1
    assert attica.nuts_id == "EL30"


def test_advanced_estimators_and_reliability():
    rng = np.random.default_rng(7)
    n = 300
    instrument = rng.normal(size=n)
    endogenous = .9 * instrument + rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)
    post = rng.integers(0, 2, size=n)
    y = 1.5 * endogenous + 2 * treatment * post + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "endog": endogenous, "z": instrument, "treat": treatment, "post": post})
    df["item1"] = rng.normal(size=n); df["item2"] = df.item1 + rng.normal(scale=.3, size=n); df["item3"] = df.item1 + rng.normal(scale=.3, size=n)
    iv_coef, iv_fit = instrumental_variables_2sls(df, "y", "endog", ["z"])
    assert abs(iv_coef.set_index("term").loc["endog", "coefficient"] - 1.5) < .3
    assert iv_fit.excluded_instrument_F.iloc[0] > 10
    did = difference_in_differences(df, "y", "treat", "post")
    assert "__did__" in set(did.coefficients.term)
    reg_coef, reg_fit = regularised_regression(df, "y", ["endog", "z"], "Ridge", 1.0)
    assert len(reg_coef) == 2 and np.isfinite(reg_fit.test_rmse.iloc[0])
    alpha, items = cronbach_alpha(df, ["item1", "item2", "item3"])
    assert alpha.cronbach_alpha.iloc[0] > .8


def test_monte_carlo_ols_is_reproducible_and_centred():
    df = synthetic(220)
    summary1, draws1, fit1 = monte_carlo_ols(df, "y", ["x1", "x2"], simulations=300, method="Wild bootstrap", seed=91)
    summary2, draws2, fit2 = monte_carlo_ols(df, "y", ["x1", "x2"], simulations=300, method="Wild bootstrap", seed=91)
    pd.testing.assert_frame_equal(draws1, draws2)
    x1 = summary1.set_index("term").loc["x1"]
    assert abs(x1.simulation_mean - 2) < .2
    assert x1.probability_positive > .99
    assert int(fit1.simulations.iloc[0]) == 300


def test_monte_carlo_portfolio_probabilities():
    df = pd.DataFrame({
        "project": ["A", "B", "C", "D"],
        "cost": [40.0, 50.0, 60.0, 80.0],
        "benefit": [100.0, 90.0, 70.0, 60.0],
    })
    summary, projects, simulations = monte_carlo_portfolio(
        df, "cost", "benefit", budget=100, project_id="project",
        simulations=300, cost_cv=.05, benefit_cv=.05, seed=17,
    )
    probs = projects.set_index("project_id").selection_probability
    assert probs["A"] > probs["D"]
    assert projects.selection_probability.between(0, 1).all()
    assert len(simulations) == 300
    assert summary.mean_portfolio_cost.iloc[0] <= 100 + 1e-8


def test_ols_and_monte_carlo_publication_bundles():
    df = synthetic(140)
    out = fit_detailed_model(df, "y", ["x1", "x2"], estimator="OLS", covariance="HC3")
    ols_zip = ols_publication_bundle(out.predictions, out.coefficients)
    assert ols_zip[:2] == b"PK"
    summary, draws, _ = monte_carlo_ols(df, "y", ["x1", "x2"], simulations=150, seed=4)
    mc_zip = monte_carlo_publication_bundle(draws, summary, "x1")
    assert mc_zip[:2] == b"PK"


def test_advanced_one_variable_clustering_and_bundle():
    rng = np.random.default_rng(100)
    absorption = np.r_[rng.normal(.25, .025, 60), rng.normal(.75, .025, 60)]
    df = pd.DataFrame({"absorption": absorption})
    out = advanced_clustering(df, ["absorption"], method="K-means", automatic_k=True, max_k=5, seed=8)
    assert out.selected_k == 2
    assert out.assignments.cluster.nunique() == 2
    assert out.diagnostics.loc[out.diagnostics.selected, "silhouette"].iloc[0] > .7
    assert out.diagnostics.loc[out.diagnostics.selected, "perturbation_stability_ari"].iloc[0] > .9
    assert clustering_publication_bundle(out.assignments, out.profiles, out.embedding, out.diagnostics)[:2] == b"PK"


def test_predictive_comparison_and_bundle():
    df = synthetic(140)
    performance, predictions, importance, comments = predictive_model_comparison(df, "y", ["x1", "x2"], folds=3, seed=5)
    assert set(["OLS", "Ridge", "Lasso", "Elastic Net", "Random forest", "Extra trees", "Gradient boosting"]) == set(performance.model)
    assert performance.iloc[0].cross_validated_r_squared > .8
    assert len(predictions) == len(df)
    assert predictive_publication_bundle(performance, importance, predictions)[:2] == b"PK"


def test_panel_model_suite_and_bundle():
    rng = np.random.default_rng(123)
    rows = []
    entity_effects = rng.normal(size=8)
    for entity in range(8):
        for year in range(2015, 2022):
            x = rng.normal() + .15 * (year - 2015)
            y = 1.4 * x + entity_effects[entity] + .08 * (year - 2015) + rng.normal(scale=.25)
            rows.append({"region": f"R{entity}", "year": year, "y": y, "x": x})
    df = pd.DataFrame(rows)
    fit, coef, hausman, prepared, comments = panel_model_suite(df, "region", "year", "y", ["x"], "Mean", "Clustered by entity")
    assert {"Pooled OLS", "Two-way fixed effects", "Random effects"}.issubset(set(fit.model))
    fe_x = coef[(coef.model == "Two-way fixed effects") & (coef.term == "x")].coefficient.iloc[0]
    assert abs(fe_x - 1.4) < .3
    assert len(prepared) == 56
    assert panel_publication_bundle(coef, fit, hausman)[:2] == b"PK"


def test_huber_and_gamma_estimators():
    df = synthetic(220)
    huber = fit_detailed_model(df, "y", ["x1", "x2"], estimator="Robust Huber")
    assert abs(huber.coefficients.set_index("term").loc["x1", "coefficient"] - 2) < .2
    gamma_df = df.assign(positive=np.exp(.3 + .25 * df.x1 + np.random.default_rng(4).normal(scale=.2, size=len(df))))
    gamma = fit_detailed_model(gamma_df, "positive", ["x1"], estimator="Gamma log-link", covariance="HC3")
    assert "exp_coefficient" in gamma.coefficients


def test_dedicated_mcda_engine_is_reproducible_and_auditable():
    alternatives = pd.DataFrame({
        "project": ["A", "B", "C", "D"],
        "benefit": [95.0, 80.0, 70.0, 55.0],
        "jobs": [30.0, 28.0, 20.0, 12.0],
        "cost": [35.0, 50.0, 65.0, 90.0],
        "risk": [.10, .18, .28, .42],
    })
    kwargs = dict(
        criteria=["benefit", "jobs", "cost", "risk"],
        directions={"benefit": "Maximise", "jobs": "Maximise", "cost": "Minimise", "risk": "Minimise"},
        weight_method="User-defined",
        user_weights={"benefit": .35, "jobs": .20, "cost": .30, "risk": .15},
        methods=["MAVT", "TOPSIS", "PROMETHEE II"],
        alternative_id="project",
        simulations=300,
        seed=19,
    )
    first = mcda_analysis(alternatives, **kwargs)
    second = mcda_analysis(alternatives, **kwargs)
    assert first.rankings.iloc[0].alternative == "A"
    assert np.isclose(first.weights.weight.sum(), 1)
    assert {"MAVT_rank", "TOPSIS_rank", "PROMETHEE II_rank", "Consensus_rank"}.issubset(first.rankings)
    pd.testing.assert_frame_equal(first.acceptability_summary, second.acceptability_summary)
    assert first.acceptability_summary.probability_rank_1.between(0, 1).all()
    assert mcda_publication_bundle(first)[:2] == b"PK"


def test_ahp_consistency_diagnostics():
    criteria = ["benefit", "cost", "risk"]
    pairwise = pd.DataFrame(
        [[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]],
        index=criteria, columns=criteria,
    )
    weights, lambda_max, consistency_ratio = ahp_weights(pairwise, criteria)
    assert np.isclose(weights.sum(), 1)
    assert weights[0] > weights[1] > weights[2]
    assert lambda_max >= 3
    assert consistency_ratio < .10


def test_research_chair_scope_formula_and_longitudinal_protocol():
    frame = pd.DataFrame({
        "year": [2019, 2020, 2021, 2022, 2023],
        "region": ["A", "A", "B", "B", "B"],
        "budget": [100.0, 120.0, 180.0, 200.0, 250.0],
        "spend": [80.0, 100.0, 135.0, 170.0, 225.0],
    })
    scoped = apply_scope(frame, ["year", "region", "budget", "spend"], "year", 2020, 2023, {"region": ["B"]})
    assert list(scoped.year) == [2021, 2022, 2023]
    enriched = add_safe_derived_column(scoped, "absorption", "spend / budget")
    assert np.allclose(enriched.absorption, [.75, .85, .90])
    result = execute_protocol(enriched, "Longitudinal trend", "absorption", ["budget"], "year", None, "Mean", r"A_t=S_t/B_t", "absorption = spend / budget")
    assert "Longitudinal results" in result.tables
    assert list(result.tables["Longitudinal results"].year.astype(int)) == [2021, 2022, 2023]


def test_research_chair_empty_model_selection_uses_all_numeric_scope():
    frame = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023],
        "budget": [100.0, 120.0, 150.0, 175.0],
        "spend": [75.0, 95.0, np.nan, 160.0],
        "region": ["A", "A", "B", "B"],
    })
    result = execute_protocol(frame, "Descriptive profile")
    assert set(result.tables["Descriptive statistics"].variable) == {"year", "budget", "spend"}
    assert len(result.tables["Variable missingness"]) == 4
    assert result.tables["Variable missingness"].iloc[0].variable == "spend"


def test_research_chair_natural_language_command_computes_results():
    frame = pd.DataFrame({
        "year": [2019, 2020, 2021, 2022, 2023],
        "budget": [100.0, 120.0, 180.0, 200.0, 250.0],
        "spend": [80.0, 100.0, 135.0, 170.0, 225.0],
        "region": ["A", "A", "B", "B", "B"],
    })
    base = execute_protocol(frame, "Descriptive profile")
    protocol = {"year_column": "year", "aggregation": "Mean", "outcome": None, "predictors": []}
    result, reply = execute_natural_language_command(
        frame, "run the analysis as in paper", protocol, base, pd.DataFrame()
    )
    assert not result.tables["Descriptive statistics"].empty
    assert not result.tables["Correlation screening"].empty
    assert not result.tables["Longitudinal results"].empty
    assert "computed results" in reply
    assert "5 records" in reply


def test_research_chair_ranks_and_explains_striking_result_safely():
    rng = np.random.default_rng(113)
    n = 180
    x = rng.normal(size=n)
    frame = pd.DataFrame({
        "project_code": np.arange(1000, 1000 + n),
        "year": np.repeat([2020, 2021, 2022], n // 3),
        "jobs_measure_a": x,
        "jobs_measure_b": 2 * x + rng.normal(scale=.05, size=n),
        "sparse_outcome": [1.0, 2.0] + [np.nan] * (n - 2),
    })
    base = execute_protocol(frame, "Descriptive profile")
    protocol = {"outcome": "year", "predictors": ["project_code"]}
    result, reply = execute_natural_language_command(
        frame, "What is the most striking statistical result?", protocol, base, pd.DataFrame()
    )
    ranked = result.tables["Ranked statistical findings"]
    assert not ranked.empty
    assert {ranked.iloc[0].variable_1, ranked.iloc[0].variable_2} == {"jobs_measure_a", "jobs_measure_b"}
    assert "project_code" not in set(ranked.variable_1) | set(ranked.variable_2)
    assert "Direct answer" in reply
    assert "Publication-ready wording" in reply
    assert "scientifically meaningless specification" in reply
    assert "98.89% missing" in reply


def test_research_chair_blocks_unsafe_expressions():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    try:
        add_safe_derived_column(frame, "bad", "__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe expression was not rejected")


def test_research_chair_pdf_selection_reply_and_bundle():
    evidence = pd.DataFrame([
        {"document": "notes.pdf", "page": 1, "text": "General introduction", "characters": 20},
        {"document": "notes.pdf", "page": 2, "text": "Regional absorption evidence", "characters": 28},
    ])
    selected = select_pdf_evidence(evidence, ["notes.pdf"], {"notes.pdf": (1, 2)}, "absorption")
    assert list(selected.page) == [2]
    data = synthetic(40)
    result = execute_protocol(data, "OLS specification", "y", ["x1", "x2"], equation=r"y_i=\beta_0+\beta_1x_{1i}+\beta_2x_{2i}+\epsilon_i")
    protocol = {"research_question": "Which factors are associated with y?", "limitations": "Observational data.", "working_title": "Test blueprint", "steps": "Estimate OLS with HC3."}
    reply = build_offline_reply("What do the regression coefficients show?", protocol, result, selected)
    assert "conditional associations" in reply
    blueprint = build_paper_blueprint(protocol, result, selected, reply)
    assert "## Research question" in blueprint and "## Limitations" in blueprint
    payload = research_bundle(data, protocol, result, selected, blueprint)
    assert payload[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        assert "OUTPUT_INTERPRETATION_GUIDE.md" in names
        assert "output_interpretation_guide.csv" in names
        assert "makryvelios_prompt_library.md" in names
        assert "makryvelios_prompt_library.csv" in names
        workbook = pd.ExcelFile(io.BytesIO(archive.read("research_command_results.xlsx")))
        assert {"Output guide", "Prompt library"}.issubset(workbook.sheet_names)


def test_prompt_library_covers_every_menu_module():
    prompts = prompt_library()
    assert len(prompts) >= 52
    assert prompts.copy_ready_prompt.str.len().gt(30).all()
    assert prompts.required_setup.str.len().gt(2).all()
    module_numbers = set(prompts.module.str.extract(r"^(\d+[A-B]?)", expand=False).dropna())
    assert {"1", "2", "3", "4", "5", "6", "6A", "7", "8", "8A", "9", "10", "10A", "10B", "11", "12", "12A", "12B", "13"}.issubset(module_numbers)


def test_output_guidance_is_plain_language_and_actionable():
    guide = build_output_guide({
        "Correlation screening": pd.DataFrame({"variable_1": ["x"], "variable_2": ["y"], "spearman_rho": [-.62], "n": [120]}),
        "OLS coefficients": pd.DataFrame({"term": ["x"], "coefficient": [.4], "p_value": [.02]}),
    }, ["Coefficient chart"])
    required = {
        "output", "format", "what_it_is", "why_it_is_used", "how_to_read_it",
        "what_the_pattern_means", "what_it_does_not_mean", "recommended_next_step",
        "plain_language_summary",
    }
    assert required.issubset(guide.columns)
    assert len(guide) == 3
    assert guide.what_it_does_not_mean.str.contains("caus", case=False).any()


def test_research_chair_extended_questions_create_downloadable_tables():
    data = synthetic(90)
    data["year"] = np.resize([2020, 2021, 2022], len(data))
    base = execute_protocol(data, "OLS specification", "y", ["x1", "x2"])
    protocol = {
        "outcome": "y", "predictors": ["x1", "x2"], "year_column": "year",
        "year_range": [2020, 2022], "equation": r"y_i=\beta_0+\beta_1x_{1i}+\beta_2x_{2i}+\epsilon_i",
    }
    requests = {
        "What can I not conclude from these results?": "Permitted and prohibited claims",
        "Write the limitations section.": "Paper limitations",
        "Which further analysis should I run?": "Further analysis roadmap",
        "Propose research questions and their answers.": "Research question proposals",
        "Give me an outline of the methodology used.": "Methodology outline",
        "Give me an outline of the results.": "Results output guide",
        "Check this equation for mathematical correctness.": "Equation and algorithm audit",
    }
    for question, expected_table in requests.items():
        result, reply = execute_natural_language_command(data, question, protocol, base, pd.DataFrame())
        assert expected_table in result.tables
        assert len(reply) > 120


def test_research_chair_monte_carlo_command_honours_seed_and_repetitions():
    data = synthetic(100)
    base = execute_protocol(data, "OLS specification", "y", ["x1", "x2"])
    protocol = {"outcome": "y", "predictors": ["x1", "x2"]}
    result, reply = execute_natural_language_command(
        data,
        "Run a wild-bootstrap Monte Carlo with 300 repetitions and seed 73.",
        protocol,
        base,
        pd.DataFrame(),
    )
    assert len(result.tables["Monte Carlo draws"]) == 300
    settings = result.tables["Monte Carlo fit and settings"].iloc[0]
    assert int(settings.simulations) == 300
    assert int(settings.seed) == 73
    assert "300" in reply and "73" in reply
