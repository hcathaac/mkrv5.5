import json
import pandas as pd
import pytest

import llm_bridge
from agentic_research import (
    AgenticPlan,
    AgenticRun,
    SYNTHESIS_RESPONSE_SCHEMA,
    _parse_synthesis_json,
    refine_run_with_ai,
)


class FakeResponse:
    def __init__(self, payload, ok=True, status_code=200, text=''):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text or json.dumps(payload)
    def json(self):
        return self._payload


def test_groq_strict_structured_synthesis_payload(monkeypatch):
    seen = {}
    payload_text = json.dumps({
        'abstract':'A', 'results':'R', 'discussion':'D', 'conclusion':'C', 'limitations':'L', 'key_findings':[]
    })
    def fake_post(url, **kwargs):
        seen['url'] = url
        seen['payload'] = kwargs['json']
        return FakeResponse({'choices':[{'message':{'content':payload_text}}]})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Groq — free plan available','api_key':'k','model':'openai/gpt-oss-120b','max_tokens':1200}
    out = llm_bridge.llm_reply('rewrite', cfg, response_schema=SYNTHESIS_RESPONSE_SCHEMA)
    assert json.loads(out)['discussion'] == 'D'
    fmt = seen['payload']['response_format']
    assert fmt['type'] == 'json_schema'
    assert fmt['json_schema']['strict'] is True
    assert fmt['json_schema']['schema']['additionalProperties'] is False
    assert seen['url'].startswith('https://api.groq.com/openai/v1')


def test_groq_error_is_not_hidden_by_provider_fallback(monkeypatch):
    calls = {'n':0}
    def fake_post(url, **kwargs):
        calls['n'] += 1
        return FakeResponse({'error':{'message':'rate limit'}}, ok=False, status_code=413, text='rate limit')
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Groq — free plan available','api_key':'k','model':'openai/gpt-oss-120b','max_tokens':1200}
    with pytest.raises(RuntimeError, match='HTTP 413'):
        llm_bridge.llm_reply('rewrite', cfg, response_schema=SYNTHESIS_RESPONSE_SCHEMA)
    assert calls['n'] == 1


def test_synthesis_parser_accepts_markdown_sectioned_prose():
    text = '''
## Abstract
This abstract names x and y and summarises the run.

## Results
x was associated with y at r=0.71.

## Discussion
The association is substantial but observational.

## Conclusion
The run supports an association between x and y.

## Limitations
No causal identification strategy was estimated.

## Key findings
- x and y show the principal association.
'''
    out = _parse_synthesis_json(text)
    assert 'x was associated' in out['results']
    assert 'observational' in out['discussion']
    assert isinstance(out['key_findings'], list)


def test_refinement_with_sectioned_prose_preserves_tables():
    run = AgenticRun(
        plan=AgenticPlan(goal='Explain y', steps=[], warnings=[], mappings={}),
        tables={'Top correlations': pd.DataFrame([{'variable_1':'x','variable_2':'y','correlation':0.71,'p_value':0.001}])},
        narratives={'discussion':'offline discussion','conclusion':'offline conclusion','limitations':'offline limitations'},
        manifest={'version':'5.8.4'},
    )
    original = run.tables['Top correlations'].copy(deep=True)
    text = '''Abstract:\nA specific abstract about x and y.\nResults:\nx and y correlated at r=0.71.\nDiscussion:\nThis is association, not causality.\nConclusion:\nThe run supports an x-y association.\nLimitations:\nNo causal design was run.\nKey findings:\n- x-y association is the principal result.'''
    refine_run_with_ai(run, lambda prompt: text, provider_label='Groq test')
    assert 'r=0.71' in run.narratives['results']
    assert run.narratives['offline_discussion'] == 'offline discussion'
    pd.testing.assert_frame_equal(original, run.tables['Top correlations'])


def test_incomplete_prose_still_fails_cleanly():
    with pytest.raises(RuntimeError, match='could not be structured'):
        _parse_synthesis_json('Discussion: only one section is present.')
