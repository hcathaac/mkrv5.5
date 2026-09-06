# Validation v5.8.5

- Python compile: PASS.
- Targeted Agentic/Groq/Gemini/Frontier/GAMS regression tests: PASS (27/27).
- New 83-variable compact-context test: PASS; prompt remains well below the prior oversized 9k+ token request.
- Full-draft synthesis respects the same selected context profile and preserves deterministic tables.
- Preservation rule: no v5.8.4 file/module/export removed.
- Provider/model routing remains manual and visible; no hidden fallback introduced.
- Existing environment caveat remains: full legacy econometrics tests requiring `linearmodels` cannot run in the current build sandbox when that optional dependency is absent, although it remains declared in deployment requirements.
