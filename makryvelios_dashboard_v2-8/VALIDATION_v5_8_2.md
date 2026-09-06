# Validation record — v5.8.2

Date: 6 September 2026

## Preservation

Version 5.8.2 is strictly additive over v5.8.1. No existing source file or analytical module was removed. The new work is limited to AI-provider routing, Agentic synthesis, documentation and tests.

## Automated tests executed in the build environment

- 19 targeted v5.8.2/v5.8.1/v5.8.0/v5.7.2 tests passed.
- Coverage included:
  - Google Gemini provider routing with mocked HTTP response;
  - Groq OpenAI-compatible provider routing with mocked HTTP response;
  - Ollama no-key provider routing with mocked HTTP response;
  - evidence-grounded Agentic synthesis and preservation of deterministic tables;
  - refined submission package generation;
  - v5.8.1 semantic Agentic conversation tests;
  - v5.8.0 frontier methods and Agentic package tests;
  - v5.7.2 GAMS/maps tests;
  - additive preservation tests.
- Python compileall completed successfully for the source tree.
- The retained `tests/test_core.py` suite passed its first 22 tests before reaching the known environment-only missing dependency `linearmodels`; that dependency remains declared in `requirements.txt` for deployment.
- Streamlit AppTest could not execute in this sandbox because Streamlit is not installed in the build environment. Streamlit remains declared in `requirements.txt` and should be exercised after deployment/reboot.

## External-provider boundary

No real API key was available in the build environment, so live Gemini/Groq/Claude calls were not sent. Provider request/response routing was tested with mocked HTTP responses. Live provider acceptance should be confirmed after the user enters a key in the deployed app.

## Scientific boundary

The AI synthesis pass is narrative-only. It does not modify deterministic result tables, regression estimates, p-values, optimisation decisions, Monte Carlo draws or source evidence. The offline first-pass narrative is retained for audit when AI synthesis succeeds.
