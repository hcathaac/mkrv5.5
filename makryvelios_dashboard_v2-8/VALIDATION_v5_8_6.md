# Makryvelios v5.8.6 validation

Validated 2026-09-06.

- Python compilation: `app.py`, `llm_bridge.py`, `agentic_research.py` passed.
- Targeted regression suite: 28 tests passed.
- Verified Groq GPT-OSS payload uses `max_completion_tokens`, visible/manual `reasoning_effort`, and `include_reasoning=false`.
- Verified strict JSON schema normalisation for nested RQ objects.
- Verified manual Medium reasoning selection is preserved (no hidden override/fallback).
- Verified empty HTTP-200 GPT-OSS responses expose finish-reason/token diagnostics.
- Retained v5.8.5 compact-context tests, v5.8.4 synthesis tests, v5.8.3 RQ reliability tests, v5.8.2 AI tests and v5.8.1 agentic-intelligence tests all passed in the targeted run.
