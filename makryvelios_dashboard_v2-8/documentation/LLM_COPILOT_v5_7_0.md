# User-Key LLM Co-pilot — v5.7.0

The LLM co-pilot is optional and additive. Every numerical, econometric, MCDA and optimisation function remains operational without an API key.

The sidebar exposes a visible session-only LLM API Key field, provider selector and model ID. Anthropic Claude is the default provider; an OpenAI-compatible endpoint can also be configured.

The API key is held only in Streamlit session state and is not written to application exports.

The LLM is never used to determine portfolio selection, regression estimates, p-values, classifications or constraints. It is invoked only after the user explicitly presses an LLM action button.

In the GAMS Studio the external model receives a compact computed-results summary for interpretation and drafting. In the Research Command Chair it receives computed result tables and selected evidence rather than silently receiving the raw uploaded workbook.

Computed output and AI interpretation are kept visually and conceptually distinct.
