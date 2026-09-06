# Deploy v5.8.4 over v5.8.3

This is a strictly additive hotfix. Replace the supplied files at their existing paths inside `makryvelios_dashboard_v2-8`, commit to `main`, then reboot the Streamlit app.

The hotfix does not introduce automatic provider/model fallback. In Agentic Research Mode, the selected provider/model remains the one actually called. Groq GPT-OSS synthesis now uses strict JSON-schema Structured Outputs when available, and the same-response parser can recover complete sectioned prose.
