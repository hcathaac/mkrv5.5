import json

import pandas as pd

import llm_bridge
from agentic_research import _parse_rq_json, generate_questions_with_ai, RQ_RESPONSE_SCHEMA


class FakeResponse:
    def __init__(self, payload, ok=True, status_code=200, text=''):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text or json.dumps(payload)
    def json(self):
        return self._payload


def test_rq_parser_accepts_fenced_json_and_wrapped_object():
    text = '''```json\n{"questions":[{"research_question":"How is x associated with y?","method_family":"OLS","variables":"x; y","priority":1,"rationale":"observed pattern","source_basis":"dataset"}]}\n```'''
    out = _parse_rq_json(text)
    assert len(out) == 1
    assert out[0]['research_question'].startswith('How is x')


def test_rq_parser_accepts_embedded_json_after_prose():
    text = 'Here are the questions you requested:\n[{"question":"Does x predict y?","method":"Predictive"}]\nHope this helps.'
    out = _parse_rq_json(text)
    assert len(out) == 1
    assert out[0]['method_family'] == 'Predictive'


def test_rq_parser_accepts_numbered_plain_text():
    text = '1. How strongly is x associated with y?\n2. Does the x-y relationship differ by region?'
    out = _parse_rq_json(text)
    assert len(out) == 2


def test_gemini_structured_output_payload(monkeypatch):
    seen = {}
    def fake_post(url, **kwargs):
        seen['payload'] = kwargs['json']
        payload = {'candidates': [{'content': {'parts': [{'text': '[{"research_question":"Does x predict y?","method_family":"Predictive","variables":"x; y","priority":2,"rationale":"test","source_basis":"data"}]'}]}}]}
        return FakeResponse(payload)
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Google Gemini — free tier available','api_key':'k','model':'gemini-3.7-flash','max_tokens':4000}
    text = llm_bridge.llm_reply('make questions', cfg, response_schema=RQ_RESPONSE_SCHEMA)
    assert 'Does x predict y?' in text
    gen = seen['payload']['generationConfig']
    assert gen['responseMimeType'] == 'application/json'
    assert gen['responseSchema']['type'] == 'array'


def test_ai_generator_falls_back_instead_of_failing_on_unparseable_output():
    df = pd.DataFrame({'x':[1,2,3,4,5,6], 'y':[2,3,5,7,11,13], 'region':['A','A','B','B','C','C']})
    calls = {'n':0}
    def bad_reply(prompt):
        calls['n'] += 1
        return 'I cannot comply with JSON formatting, but here is some prose without a question mark.'
    out = generate_questions_with_ai(df, None, 'Explain relationships', 10, bad_reply, batch_size=5)
    assert len(out) == 10
    assert out.attrs['ai_generated'] == 0
    assert out.attrs['deterministic_recovery'] == 10
    assert out.attrs['parse_failures'] >= 2


def test_ai_generator_keeps_plain_text_questions_from_model():
    df = pd.DataFrame({'x':[1,2,3,4,5,6], 'y':[2,3,5,7,11,13]})
    def reply(prompt):
        return '1. How strongly is x associated with y?\n2. Does x improve prediction of y?'
    out = generate_questions_with_ai(df, None, 'Explain relationships', 2, reply, batch_size=2)
    assert len(out) == 2
    assert out.attrs['ai_generated'] == 2

def test_gemini_structured_output_retries_plain_if_schema_rejected(monkeypatch):
    calls = []
    def fake_post(url, **kwargs):
        calls.append(kwargs['json'])
        if len(calls) == 1:
            return FakeResponse({'error': {'message':'schema unsupported'}}, ok=False, status_code=400, text='schema unsupported')
        return FakeResponse({'candidates': [{'content': {'parts': [{'text': '1. How is x associated with y?'}]}}]})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Google Gemini — free tier available','api_key':'k','model':'gemini-3.7-flash','max_tokens':4000}
    text = llm_bridge.llm_reply('make questions', cfg, response_schema=RQ_RESPONSE_SCHEMA)
    assert 'How is x associated with y?' in text
    assert len(calls) == 2
    assert 'responseMimeType' in calls[0]['generationConfig']
    assert 'responseMimeType' not in calls[1]['generationConfig']
