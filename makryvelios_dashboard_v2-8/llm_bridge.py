"""Optional LLM bridge for interpretation, synthesis and drafting.

Numerical analysis remains deterministic and local to the application. External
LLMs receive only the evidence text explicitly supplied by the calling workflow.
API keys are held in Streamlit session state and are never written to exports.

Supported routes:
- Anthropic Claude
- Google Gemini (free tier available for selected models)
- Groq (free plan available; OpenAI-compatible chat endpoint)
- Ollama Local (no API key)
- generic OpenAI-compatible endpoints
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import requests


@dataclass
class LLMConfig:
    provider: str = "Anthropic Claude"
    api_key: str = ""
    model: str = "claude-sonnet-5"
    base_url: str = ""
    max_tokens: int = 2500
    reasoning_effort: str = "low"


def configured(config: Mapping[str, Any] | LLMConfig | None) -> bool:
    if config is None:
        return False
    cfg = _as_config(config) if not isinstance(config, LLMConfig) else config
    provider = cfg.provider.strip().lower()
    if provider.startswith("ollama"):
        return bool(cfg.model.strip())
    return bool(cfg.api_key.strip() and cfg.model.strip())


def _as_config(config: Mapping[str, Any] | LLMConfig) -> LLMConfig:
    if isinstance(config, LLMConfig):
        return config
    return LLMConfig(
        provider=str(config.get("provider", "Anthropic Claude")),
        api_key=str(config.get("api_key", "")),
        model=str(config.get("model", "claude-sonnet-5")),
        base_url=str(config.get("base_url", "")),
        max_tokens=int(config.get("max_tokens", 2500)),
        reasoning_effort=str(config.get("reasoning_effort", "low")),
    )


def llm_reply(
    prompt: str,
    config: Mapping[str, Any] | LLMConfig,
    *,
    system: str = "You are a scientific research assistant. Preserve numerical evidence exactly, distinguish computed results from interpretation, and never invent missing findings.",
    timeout: int = 90,
    response_schema: Mapping[str, Any] | None = None,
) -> str:
    cfg = _as_config(config)
    if not configured(cfg):
        raise ValueError("Configure an AI model first. Hosted providers require an API key; Ollama Local does not.")
    provider = cfg.provider.strip().lower()
    if provider.startswith("anthropic"):
        return _anthropic_reply(prompt, cfg, system=system, timeout=timeout)
    if provider.startswith("google") or provider.startswith("gemini"):
        return _gemini_reply(prompt, cfg, system=system, timeout=timeout, response_schema=response_schema)
    if provider.startswith("groq"):
        if not cfg.base_url.strip():
            cfg.base_url = "https://api.groq.com/openai/v1"
        return _openai_compatible_reply(prompt, cfg, system=system, timeout=timeout, response_schema=response_schema, strict_json_schema=True)
    if provider.startswith("ollama"):
        return _ollama_reply(prompt, cfg, system=system, timeout=timeout)
    if provider.startswith("openai") or "compatible" in provider:
        return _openai_compatible_reply(prompt, cfg, system=system, timeout=timeout)
    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")


def _anthropic_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": cfg.model,
            "max_tokens": max(256, min(int(cfg.max_tokens), 12000)),
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    if not response.ok:
        detail = response.text[:1200]
        raise RuntimeError(f"Anthropic API returned HTTP {response.status_code}: {detail}")
    payload = response.json()
    pieces = []
    for item in payload.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            pieces.append(str(item.get("text", "")))
    text = "\n".join(piece for piece in pieces if piece).strip()
    if not text:
        raise RuntimeError("Anthropic API returned no text content.")
    return text


def _gemini_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int, response_schema: Mapping[str, Any] | None = None) -> str:
    model = cfg.model.strip() or "gemini-3.7-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def make_payload(schema: Mapping[str, Any] | None) -> dict:
        generation = {
            "maxOutputTokens": max(256, min(int(cfg.max_tokens), 12000)),
            "temperature": 0.18,
        }
        if schema:
            generation.update({"responseMimeType": "application/json", "responseSchema": dict(schema)})
        return {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": generation,
        }

    headers = {"x-goog-api-key": cfg.api_key, "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=make_payload(response_schema), timeout=timeout)
    # Structured outputs are preferred for agentic workflows. If a particular
    # model/API revision rejects the schema fields, retry once as ordinary text
    # so the tolerant downstream parser can still recover useful content.
    if not response.ok and response_schema is not None and int(response.status_code) in {400, 404, 422}:
        response = requests.post(endpoint, headers=headers, json=make_payload(None), timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {response.text[:1200]}")
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        feedback = payload.get("promptFeedback", {})
        raise RuntimeError(f"Gemini API returned no candidates. {feedback}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")).strip()
    if not text:
        raise RuntimeError("Gemini API returned no text content.")
    return text

def _ollama_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int) -> str:
    base = cfg.base_url.strip().rstrip("/") or "http://127.0.0.1:11434"
    endpoint = base + "/api/generate"
    response = requests.post(
        endpoint,
        json={
            "model": cfg.model,
            "prompt": f"SYSTEM\n{system}\n\nUSER\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.12, "num_predict": max(256, min(int(cfg.max_tokens), 12000))},
        },
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text[:1200]}")
    text = str(response.json().get("response", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned no response text.")
    return text


def _strict_schema_for_groq(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Groq strict-JSON-schema compatible copy.

    Groq strict structured outputs require every object property to be required
    and ``additionalProperties`` to be false.  The application schemas already
    use explicit fields; this normaliser makes nested/array schemas safe without
    changing the semantic contract.
    """
    def normalise(node: Any) -> Any:
        if isinstance(node, list):
            return [normalise(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {k: normalise(v) for k, v in node.items()}
        typ = out.get("type")
        if typ == "object" or "properties" in out:
            props = out.get("properties", {})
            if isinstance(props, dict):
                out["properties"] = {k: normalise(v) for k, v in props.items()}
                out["required"] = list(props.keys())
                out["additionalProperties"] = False
        if "items" in out:
            out["items"] = normalise(out["items"])
        return out
    return normalise(dict(schema))


def _openai_compatible_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int, response_schema: Mapping[str, Any] | None = None, strict_json_schema: bool = False) -> str:
    base = cfg.base_url.strip().rstrip("/") or "https://api.openai.com/v1"
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    is_groq = "api.groq.com" in base.lower() or cfg.provider.strip().lower().startswith("groq")
    is_gpt_oss = is_groq and cfg.model.strip().lower().startswith("openai/gpt-oss-")

    # Groq recommends placing GPT-OSS instructions in the user turn and exposes
    # a separate reasoning budget.  Low reasoning is the correct default for
    # extraction/structured-output tasks: it preserves completion budget for the
    # actual JSON instead of spending it on internal reasoning.
    messages = (
        [{"role": "user", "content": f"INSTRUCTIONS\n{system}\n\nTASK\n{prompt}"}]
        if is_gpt_oss else
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    )
    completion_limit = max(256, min(int(cfg.max_tokens), 12000))
    payload: dict[str, Any] = {
        "model": cfg.model,
        "temperature": 0.15,
        "messages": messages,
    }
    if is_gpt_oss:
        payload["max_completion_tokens"] = completion_limit
        effort = str(getattr(cfg, "reasoning_effort", "low") or "low").strip().lower()
        payload["reasoning_effort"] = effort if effort in {"low", "medium", "high"} else "low"
        payload["include_reasoning"] = False
    else:
        payload["max_tokens"] = completion_limit

    if response_schema is not None:
        schema = _strict_schema_for_groq(response_schema) if is_groq and strict_json_schema else dict(response_schema)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "makryvelios_structured_response",
                "strict": bool(strict_json_schema),
                "schema": schema,
            },
        }

    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI-compatible API returned HTTP {response.status_code}: {response.text[:1200]}")
    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI-compatible API returned no choices.")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text"))
    text = str(content or "").strip()
    if not text:
        finish = str(choice.get("finish_reason", "unknown"))
        usage = body.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", "unknown"))
        reasoning_present = bool(message.get("reasoning"))
        model_note = (
            f" Groq GPT-OSS reasoning_effort={payload.get('reasoning_effort')} and "
            f"max_completion_tokens={payload.get('max_completion_tokens')}."
            if is_gpt_oss else ""
        )
        raise RuntimeError(
            "OpenAI-compatible API returned HTTP 200 but no final text content "
            f"(finish_reason={finish}; completion_tokens={completion_tokens}; reasoning_present={reasoning_present})."
            + model_note
            + " This is a generation-budget/response-state issue, not an authentication failure."
        )
    return text


def summarise_ita_for_llm(run, *, max_projects: int = 40) -> str:
    """Create a compact evidence-only context from a GAMS-compatible run."""
    project_results = run.project_results.copy()
    selected = project_results.loc[project_results.selected.eq(1)].sort_values("weighted_score", ascending=False)
    top = selected.head(max_projects)
    evidence = {
        "solver_status": run.status,
        "solver": run.settings.get("solver"),
        "portfolio_score": run.objective,
        "selected_projects": int(project_results.selected.sum()),
        "allocated_budget": float(project_results.allocated_budget.sum()),
        "top_selected_projects": top[["project_id", "weighted_score", "effective_budget"]].to_dict("records"),
        "region_allocation": run.region_allocation.to_dict("records"),
        "sector_allocation": run.sector_allocation.to_dict("records"),
        "intervention_allocation": run.intervention_allocation.to_dict("records"),
        "most_binding_constraints": run.constraint_diagnostics.sort_values("utilisation", ascending=False).head(15).to_dict("records") if not run.constraint_diagnostics.empty else [],
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
