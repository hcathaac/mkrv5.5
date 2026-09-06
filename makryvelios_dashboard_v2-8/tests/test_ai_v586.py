import json
import llm_bridge


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)
    def json(self):
        return self._payload


def test_groq_gpt_oss_uses_low_reasoning_and_completion_budget(monkeypatch):
    captured = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        captured['url'] = url
        captured['payload'] = json
        return FakeResponse({
            'choices': [{'message': {'content': '[{"research_question":"Q?","method_family":"x","variables":"x","priority":1,"rationale":"r","source_basis":"s"}]'}, 'finish_reason':'stop'}],
            'usage': {'completion_tokens': 150}
        })
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    schema = {
        'type':'array', 'items': {'type':'object', 'properties': {
            'research_question': {'type':'string'}, 'method_family':{'type':'string'}, 'variables':{'type':'string'},
            'priority':{'type':'integer'}, 'rationale':{'type':'string'}, 'source_basis':{'type':'string'},
        }, 'required':['research_question','method_family','variables','priority','rationale','source_basis']}
    }
    cfg = {'provider':'Groq — free plan available','api_key':'x','model':'openai/gpt-oss-120b','base_url':'https://api.groq.com/openai/v1','max_tokens':1440,'reasoning_effort':'low'}
    out = llm_bridge.llm_reply('prompt', cfg, response_schema=schema)
    assert 'research_question' in out
    p = captured['payload']
    assert p['max_completion_tokens'] == 1440
    assert 'max_tokens' not in p
    assert p['reasoning_effort'] == 'low'
    assert p['include_reasoning'] is False
    assert len(p['messages']) == 1 and p['messages'][0]['role'] == 'user'
    item = p['response_format']['json_schema']['schema']['items']
    assert item['additionalProperties'] is False
    assert set(item['required']) == set(item['properties'])


def test_groq_reasoning_effort_manual_medium(monkeypatch):
    captured = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        captured['payload'] = json
        return FakeResponse({'choices':[{'message':{'content':'OK'},'finish_reason':'stop'}]})
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Groq — free plan available','api_key':'x','model':'openai/gpt-oss-120b','base_url':'https://api.groq.com/openai/v1','max_tokens':800,'reasoning_effort':'medium'}
    assert llm_bridge.llm_reply('prompt', cfg) == 'OK'
    assert captured['payload']['reasoning_effort'] == 'medium'


def test_empty_groq_content_has_diagnostic(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse({
            'choices':[{'message':{'content':None, 'reasoning':'internal reasoning only'}, 'finish_reason':'length'}],
            'usage': {'completion_tokens': 1440}
        })
    monkeypatch.setattr(llm_bridge.requests, 'post', fake_post)
    cfg = {'provider':'Groq — free plan available','api_key':'x','model':'openai/gpt-oss-120b','base_url':'https://api.groq.com/openai/v1','max_tokens':1440,'reasoning_effort':'low'}
    try:
        llm_bridge.llm_reply('prompt', cfg)
    except RuntimeError as exc:
        msg = str(exc)
        assert 'HTTP 200 but no final text content' in msg
        assert 'finish_reason=length' in msg
        assert 'reasoning_present=True' in msg
        assert 'reasoning_effort=low' in msg
    else:
        raise AssertionError('Expected diagnostic error')
