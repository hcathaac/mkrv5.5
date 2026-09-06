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
    )


def llm_reply(
    prompt: str,
    config: Mapping[str, Any] | LLMConfig,
    *,
    system: str = "You are a scientific research assistant. Preserve numerical evidence exactly, distinguish computed results from interpretation, and never invent missing findings.",
    timeout: int = 90,
) -> str:
    cfg = _as_config(config)
    if not configured(cfg):
        raise ValueError("Configure an AI model first. Hosted providers require an API key; Ollama Local does not.")
    provider = cfg.provider.strip().lower()
    if provider.startswith("anthropic"):
        return _anthropic_reply(prompt, cfg, system=system, timeout=timeout)
    if provider.startswith("google") or provider.startswith("gemini"):
        return _gemini_reply(prompt, cfg, system=system, timeout=timeout)
    if provider.startswith("groq"):
        if not cfg.base_url.strip():
            cfg.base_url = "https://api.groq.com/openai/v1"
        return _openai_compatible_reply(prompt, cfg, system=system, timeout=timeout)
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


def _gemini_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int) -> str:
    model = cfg.model.strip() or "gemini-3.7-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": cfg.api_key, "Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {
                "maxOutputTokens": max(256, min(int(cfg.max_tokens), 12000)),
                "temperature": 0.18,
            },
        },
        timeout=timeout,
    )
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


def _openai_compatible_reply(prompt: str, cfg: LLMConfig, *, system: str, timeout: int) -> str:
    base = cfg.base_url.strip().rstrip("/") or "https://api.openai.com/v1"
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        json={
            "model": cfg.model,
            "max_tokens": max(256, min(int(cfg.max_tokens), 12000)),
            "temperature": 0.15,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI-compatible API returned HTTP {response.status_code}: {response.text[:1200]}")
    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI-compatible API returned no choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    text = str(content).strip()
    if not text:
        raise RuntimeError("OpenAI-compatible API returned no text content.")
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
