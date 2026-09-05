from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from frontier_methods import (
    bayesian_linear_regression,
    causal_aipw,
    explainable_random_forest,
    pareto_portfolio,
)
from agentic_research import (
    agentic_submission_package,
    build_agentic_plan,
    extract_source_evidence,
    generate_research_questions,
    offline_agent_reply,
    run_agentic_workflow,
)


def synthetic_frame(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(580)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(.55 * x1 - .25 * x2)))
    treatment = rng.binomial(1, p)
    y = 1.75 * treatment + 1.2 * x1 - .6 * x2 + rng.normal(scale=.8, size=n)
    return pd.DataFrame({
        "project_id": [f"P{i:03d}" for i in range(n)],
        "cost": rng.uniform(1, 12, n),
        "obj1": rng.uniform(0, 5, n),
        "obj2": rng.uniform(0, 5, n),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "treatment": treatment,
        "outcome": y,
        "group": np.where(x3 > 0, "A", "B"),
    })


def test_pareto_robust_portfolio():
    df = synthetic_frame(80)
    result = pareto_portfolio(df, project_id="project_id", objectives=["obj1", "obj2"], cost_column="cost", budget=float(df.cost.sum() * .35), resolution=4)
    assert not result.frontier.empty
    assert {"total_obj1", "total_obj2", "effective_cost"}.issubset(result.frontier.columns)
    assert result.project_frequency.pareto_selection_frequency.between(0, 1).all()


def test_cross_fitted_aipw_recovers_signal():
    df = synthetic_frame(260)
    result = causal_aipw(df, outcome="outcome", treatment="treatment", covariates=["x1", "x2", "x3"], folds=5)
    ate = float(result.estimate.estimate.iloc[0])
    assert 1.0 < ate < 2.5
    assert not result.balance.empty
    assert result.estimate["overlap_share_0.05_0.95"].iloc[0] > .8


def test_bayesian_posterior_and_predictive():
    df = synthetic_frame(180)
    result = bayesian_linear_regression(df, outcome="outcome", predictors=["x1", "x2", "x3"], draws=600, seed=580)
    assert len(result.draws) == 600
    row = result.summary[result.summary.term == "x1"].iloc[0]
    assert row["P(beta>0)"] > .95
    assert result.diagnostics["posterior_predictive_95pct_coverage"].between(0, 1).all()


def test_shap_or_fallback_explainability():
    df = synthetic_frame(140)
    result = explainable_random_forest(df, target="outcome", features=["x1", "x2", "x3"])
    assert not result.global_importance.empty
    assert not result.local_explanation.empty
    assert result.backend


def test_agentic_150_questions_and_package_without_llm():
    df = synthetic_frame(160)
    pages = pd.DataFrame([
        {"document": "paper1.pdf", "page": 1, "text": "Regional innovation and project selection were analysed using regression and optimisation. doi:10.1000/example1", "characters": 120},
        {"document": "paper1.pdf", "page": 2, "text": "The results discuss robust allocation, uncertainty and public investment priorities in 2024.", "characters": 100},
    ])
    evidence = extract_source_evidence(pages)
    assert evidence.doi.str.contains("10.1000/example1", regex=False).any()
    rqs = generate_research_questions(df, pages, limit=150)
    assert len(rqs) == 150
    plan = build_agentic_plan(df, goal="Test complete research runner", outcome="outcome", predictors=["x1", "x2", "x3"], group="group", pdf_pages=pages)
    run = run_agentic_workflow(df, plan=plan, pdf_pages=pages, outcome="outcome", predictors=["x1", "x2", "x3"], group="group", question_limit=150)
    assert len(run.tables["Research questions"]) == 150
    offline_answer = offline_agent_reply(run, "What is the strongest finding and conclusion?")
    assert isinstance(offline_answer, str) and len(offline_answer) > 40
    payload = agentic_submission_package(run, title="Test paper")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = set(z.namelist())
    required = {
        "paper/paper_draft.docx",
        "results/complete_results.xlsx",
        "results/manifest.json",
        "report/interactive_report.html",
        "figures/correlation_matrix.png",
        "figures/correlation_matrix.svg",
        "figures/correlation_matrix.pdf",
    }
    assert required.issubset(names)
