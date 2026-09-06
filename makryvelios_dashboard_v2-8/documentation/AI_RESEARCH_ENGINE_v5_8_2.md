# AI / LLM Research Engine — v5.8.2

Version 5.8.2 expands the optional language-model layer without changing any deterministic analytical engine.

## Providers

The persistent sidebar panel supports:

- Anthropic Claude;
- Google Gemini (free tier available for selected models);
- Groq (free plan available, OpenAI-compatible endpoint);
- Ollama Local (no API key);
- a generic OpenAI-compatible endpoint.

API keys are stored only in Streamlit session state and are not written to exports. Ollama Local requires no API key.

## Agentic synthesis pass

12D Agentic Research Mode first executes the approved deterministic workflow. If an AI engine is selected, the application can then perform a second-pass evidence synthesis using retrieved result rows and uploaded-PDF page evidence. The AI pass may rewrite:

- Abstract;
- Results;
- Discussion;
- Conclusion;
- run-specific Limitations;
- a structured key-findings table.

The AI pass never changes regression estimates, p-values, optimisation decisions, Monte Carlo draws, source tables or any other computed result. The deterministic first-pass narrative is retained for audit.

## Free and local routes

Gemini and Groq may provide free usage subject to provider quotas and terms. Ollama is the privacy-first local/no-key route when the app and model are reachable on the same host/network.

For Streamlit Community Cloud, `127.0.0.1` is the cloud container, not the user's laptop. A laptop-hosted Ollama server therefore requires an explicitly reachable endpoint or local deployment of the Streamlit application.

## Privacy

The UI explicitly warns that free-tier hosted providers may have data-use terms that are unsuitable for confidential research. Users should review provider terms before sending restricted material. The application never sends raw data automatically: AI actions are user-triggered and use bounded evidence context produced by the deterministic workflow.
