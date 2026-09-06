import json
from types import SimpleNamespace

import pandas as pd

import llm_bridge
from agentic_research import AgenticPlan, AgenticRun, refine_run_with_ai


class FakeResponse:
    def __init__(self, payload, ok=True, status_code=200, text=''):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text or json.dumps(payload)
    def json(self):
        return self._payload
    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.text)


def test_gemini_provider_route(monkeypatch):
    seen = {}
    def fake_post(url, **kwargs):
        seen['url'] = url
        seen['headers'] = kwargs.get('headers', {})
        return FakeResponse({'candidates': [{'content': {'parts': [{'text': 'grounded answer'}]}}]})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider': 'Google Gemini — free tier available', 'api_key': 'k', 'model': 'gemini-3.7-flash', 'max_tokens': 1000}
    assert llm_bridge.llm_reply('hello', cfg) == 'grounded answer'
    assert 'generativelanguage.googleapis.com' in seen['url']
    assert seen['headers']['x-goog-api-key'] == 'k'


def test_groq_provider_route(monkeypatch):
    seen = {}
    def fake_post(url, **kwargs):
        seen['url'] = url
        return FakeResponse({'choices': [{'message': {'content': 'fast answer'}}]})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider': 'Groq — free plan available', 'api_key': 'k', 'model': 'openai/gpt-oss-120b', 'max_tokens': 1000}
    assert llm_bridge.llm_reply('hello', cfg) == 'fast answer'
    assert seen['url'].startswith('https://api.groq.com/openai/v1')


def test_ollama_requires_no_key(monkeypatch):
    def fake_post(url, **kwargs):
        return FakeResponse({'response': 'local answer'})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider': 'Ollama Local — no API key', 'api_key': '', 'model': 'qwen3:8b', 'base_url': 'http://127.0.0.1:11434'}
    assert llm_bridge.configured(cfg)
    assert llm_bridge.llm_reply('hello', cfg) == 'local answer'


def test_ai_refinement_rewrites_specific_sections_without_touching_tables():
    plan = AgenticPlan(goal='Explain outcome variation', steps=[], warnings=[], mappings={})
    run = AgenticRun(
        plan=plan,
        tables={
            'OLS coefficients': pd.DataFrame([{
                'term': 'x1', 'coefficient': 2.1, 'ci_95_low': 1.2, 'ci_95_high': 3.0, 'p_value': 0.001
            }]),
            'Top correlations': pd.DataFrame([{
                'variable_1': 'x1', 'variable_2': 'outcome', 'correlation': 0.71, 'p_value': 0.001
            }]),
        },
        narratives={'discussion': 'generic offline', 'conclusion': 'generic conclusion', 'limitations': 'generic limitations'},
        manifest={'version': '5.8.2'},
    )
    original = run.tables['OLS coefficients'].copy(deep=True)
    payload = {
        'abstract': 'The analysis found a strong adjusted association between x1 and outcome.',
        'results': 'x1 had beta 2.1 with 95% CI 1.2 to 3.0 and p=0.001.',
        'discussion': 'The x1 estimate is the principal model result and should be interpreted as adjusted association.',
        'conclusion': 'Within this model, higher x1 is associated with higher outcome.',
        'limitations': 'The run used OLS and does not establish causal identification.',
        'key_findings': [{'finding':'x1 association','evidence':'beta 2.1','strength':'strong','safe_inference':'adjusted association'}],
    }
    refine_run_with_ai(run, lambda prompt: json.dumps(payload), provider_label='Test AI')
    assert 'x1' in run.narratives['results']
    assert run.narratives['offline_discussion'] == 'generic offline'
    pd.testing.assert_frame_equal(original, run.tables['OLS coefficients'])
    assert run.manifest['ai_synthesis']['provider'] == 'Test AI'
    assert 'AI synthesis key findings' in run.tables


def test_refined_submission_package_contains_docx_and_manifest():
    from agentic_research import agentic_submission_package
    import zipfile, io
    plan = AgenticPlan(goal='Explain outcome variation', steps=[], warnings=[], mappings={})
    run = AgenticRun(
        plan=plan,
        tables={'Top correlations': pd.DataFrame([{'variable_1':'x1','variable_2':'y','correlation':0.7,'p_value':0.001}])},
        narratives={'discussion':'offline','conclusion':'offline conclusion','limitations':'offline limits'},
        manifest={'version':'5.8.2'},
    )
    payload = {
        'abstract':'Specific abstract about x1 and y.',
        'results':'x1 and y were correlated at r=0.7, p=0.001.',
        'discussion':'The x1-y association is substantial but not causal.',
        'conclusion':'The run supports an association between x1 and y.',
        'limitations':'No causal identification strategy was estimated.',
        'key_findings':[],
    }
    refine_run_with_ai(run, lambda prompt: json.dumps(payload), provider_label='Test AI')
    blob = agentic_submission_package(run, title='Test refined paper')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = set(z.namelist())
        assert 'paper/paper_draft.docx' in names
        manifest = json.loads(z.read('results/manifest.json'))
        assert manifest['ai_synthesis']['provider'] == 'Test AI'
        md = z.read('paper/paper_draft.md').decode('utf-8')
        assert 'x1 and y were correlated' in md
