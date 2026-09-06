import numpy as np
import pandas as pd

from agentic_research import (
    AgenticPlan,
    AgenticRun,
    ai_research_question_prompt,
    estimate_rq_prompt_tokens,
    generate_questions_with_ai,
    refine_run_with_ai,
)


def _wide_df(rows=200):
    rng = np.random.default_rng(42)
    data = {f"x{i:02d}": rng.normal(size=rows) for i in range(75)}
    data.update({
        "Project_start_year": rng.integers(2007, 2026, size=rows),
        "Project_end_year": rng.integers(2010, 2027, size=rows),
        "Final_Project_Budget": rng.lognormal(12, 0.8, size=rows),
        "Region": rng.choice(["ATT", "CMK", "STE", "WMK"], size=rows),
        "Sector": rng.choice(["1", "2", "3"], size=rows),
        "Intervention": rng.choice(["1", "2"], size=rows),
        "Project_score": rng.uniform(0, 5, size=rows),
        "Funded": rng.integers(0, 2, size=rows),
    })
    return pd.DataFrame(data)


def test_compact_rq_prompt_stays_small_for_83_variable_dataset():
    df = _wide_df()
    focus = ["Project_start_year", "Project_end_year", "Final_Project_Budget", "Region"]
    prompt = ai_research_question_prompt(
        df, None, "Explain the strongest defensible relationships and propose publication-ready questions.", 10,
        focus_columns=focus, context_profile="Compact",
    )
    estimate = estimate_rq_prompt_tokens(
        df, None, "Explain the strongest defensible relationships and propose publication-ready questions.", 10,
        focus_columns=focus, context_profile="Compact",
    )
    assert len(prompt) < 15000
    assert estimate < 3500
    for column in focus:
        assert column in prompt


def test_compact_ai_generator_uses_focus_and_returns_questions():
    df = _wide_df(80)
    seen = {"prompt": ""}
    def reply(prompt):
        seen["prompt"] = prompt
        return '[{"research_question":"How is Final_Project_Budget associated with Project_start_year?","method_family":"Econometric","variables":"Final_Project_Budget; Project_start_year","priority":1,"rationale":"observed relationship","source_basis":"active dataset"}]'
    out = generate_questions_with_ai(
        df, None, "Explain budget patterns", 1, reply, batch_size=1,
        focus_columns=["Final_Project_Budget", "Project_start_year"], context_profile="Compact",
    )
    assert len(out) == 1
    assert "Final_Project_Budget" in seen["prompt"]
    assert "COMPACT SCHEMA" in seen["prompt"]


def test_synthesis_compact_profile_limits_context_without_changing_tables():
    big = pd.DataFrame({"variable_1":["x"]*200, "variable_2":["y"]*200, "correlation":[0.7]*200, "p_value":[0.001]*200})
    run = AgenticRun(
        plan=AgenticPlan(goal="Explain x and y", steps=[], warnings=[], mappings={}),
        tables={"Top correlations": big},
        narratives={"discussion":"old D", "conclusion":"old C", "limitations":"old L"},
        manifest={},
    )
    seen = {"n":0}
    def reply(prompt):
        seen["n"] = len(prompt)
        return '{"abstract":"A specific abstract about x and y.","results":"x and y are associated at r=0.7.","discussion":"The observed association is substantial but not causal.","conclusion":"The run supports an x-y association.","limitations":"No causal identification design was estimated.","key_findings":[]}'
    before = run.tables["Top correlations"].copy(deep=True)
    refine_run_with_ai(run, reply, provider_label="test", context_profile="Compact")
    assert seen["n"] < 22000
    pd.testing.assert_frame_equal(before, run.tables["Top correlations"])
