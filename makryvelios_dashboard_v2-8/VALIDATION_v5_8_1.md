# v5.8.1 Validation Record

Date: 2026-09-06

## Scope

Strictly additive Agentic intelligence upgrade over v5.8.0. No pre-existing analytical, econometric, Frontier, ITA/GAMS, mapping, respondent, Research Chair, visualisation or export engine is removed.

## Preservation

- Removed files relative to v5.8.0: **0**.
- Changes are limited to Agentic intelligence/UI, version/header wiring, documentation and tests.
- Existing numerical engines remain unchanged; Agentic AI remains an interpretation/planning layer over deterministic outputs.

## Functional validation

- 14 targeted automated tests passed: new v5.8.1 Agentic semantic/local-AI tests plus retained Frontier, GAMS-map and preservation suites.
- Exact screenshot-style question tested: `What is the weakest finding and what I cannot safely conclude?` returns the least-supported substantive OLS term with beta, 95% CI and p-value from the computed run rather than a generic limitations paragraph.
- Named-model-term question tested: a direct question about a predictor returns that predictor's exact configured OLS coefficient/CI/p-value.
- Semantic retrieval tested over actual run tables/narratives.
- Data-aware offline RQ generation tested with rationale and source basis.
- Local Ollama bridge tested with a mocked HTTP response and verified to require no API key.
- Top-level Python compilation: **22 files compiled successfully**.

## Retained core-suite environment note

The retained `tests/test_core.py` executed 22 tests successfully before reaching the existing panel-model test, which cannot run in this build sandbox because `linearmodels` is not installed here. `linearmodels` remains declared in the deployment requirements. This is the same environment limitation documented in the earlier releases and is unrelated to v5.8.1.

## Local AI deployment note

Ollama is optional and accessed by HTTP; no new Python dependency is mandatory. When Streamlit runs locally on the same computer/server, the normal endpoint is `http://127.0.0.1:11434`. Streamlit Community Cloud cannot directly access an end user's laptop localhost; a network-reachable/self-hosted Ollama endpoint is required for local-model intelligence from the cloud deployment.
