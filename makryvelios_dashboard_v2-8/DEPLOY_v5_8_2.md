# Deploy v5.8.2 over v5.8.1

This is a strictly additive hotfix. Keep the existing repository, branch and Streamlit entry point unchanged. Replace the supplied files at their existing relative paths and reboot the Streamlit app.

## New visible behaviour

- Sidebar **AI / LLM RESEARCH ENGINE** provider list: Claude, Google Gemini free tier, Groq free plan, Ollama Local (no key), OpenAI-compatible.
- **12D Agentic Research Mode** automatically selects the configured sidebar AI when available.
- Optional automatic AI synthesis after deterministic execution.
- One-click **REFINE / REWRITE ENTIRE DRAFT WITH SELECTED AI**.
- AI-refined Abstract, Results, Discussion, Conclusion and run-specific Limitations are used by the submission package.
- Deterministic numerical results are never modified by AI.

## Privacy

Gemini free-tier use is explicitly marked as unsuitable for confidential/restricted material unless its data-use terms are acceptable to the project. Ollama remains the no-key/local route.
