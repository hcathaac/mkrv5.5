from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_research import (
    build_agentic_plan,
    generate_research_questions,
    offline_agent_reply,
    run_agentic_workflow,
    semantic_retrieve_run,
)


def _run():
    rng = np.random.default_rng(581)
    n = 140
    strong = rng.normal(size=n)
    weak = rng.normal(size=n)
    outcome = 1.9 * strong + 0.01 * weak + rng.normal(scale=.65, size=n)
    df = pd.DataFrame({
        "outcome": outcome,
        "strong_predictor": strong,
        "weak_predictor": weak,
        "group": np.where(rng.random(n) > .5, "A", "B"),
        "year": rng.integers(2020, 2026, n),
    })
    plan = build_agentic_plan(df, goal="Identify strong and weak evidence", outcome="outcome", predictors=["strong_predictor", "weak_predictor"], group="group", time_column="year")
    return df, run_agentic_workflow(df, plan=plan, outcome="outcome", predictors=["strong_predictor", "weak_predictor"], group="group", question_limit=40)


def test_exact_weakest_question_is_specific():
    _, run = _run()
    answer = offline_agent_reply(run, "What is the weakest finding and what I cannot safely conclude?")
    assert "weak_predictor" in answer
    assert "β=" in answer and "95% CI" in answer and "p=" in answer
    assert "caus" in answer.lower()


def test_named_term_question_retrieves_model_row():
    _, run = _run()
    answer = offline_agent_reply(run, "What does weak_predictor tell me in this model?")
    assert "weak_predictor" in answer
    assert "β=" in answer and "p=" in answer


def test_semantic_retrieval_returns_run_evidence():
    _, run = _run()
    hits = semantic_retrieve_run(run, "weak predictor coefficient model", top_k=5)
    assert hits
    assert any("OLS" in h["source"] or "weak_predictor" in h["text"] for h in hits)


def test_offline_rqs_are_data_aware():
    df, _ = _run()
    rqs = generate_research_questions(df, None, 40)
    assert len(rqs) == 40
    assert {"rationale", "source_basis"}.issubset(rqs.columns)
    text = " ".join(rqs.research_question.astype(str))
    assert "strong_predictor" in text and "outcome" in text


def test_ollama_bridge_requires_no_api_key(monkeypatch):
    from agentic_research import ollama_text_reply

    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"response": "local grounded answer"}

    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert "api_key" not in json
        assert json["model"] == "test-local"
        return Response()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    answer = ollama_text_reply("question", "test-local", endpoint="http://127.0.0.1:11434")
    assert answer == "local grounded answer"
