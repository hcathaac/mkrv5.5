# Deploy v5.8.1 over v5.8.0

Version 5.8.1 is strictly additive. Replace files at their existing relative paths inside `makryvelios_dashboard_v2-8`; do not create a nested release directory.

Main file path remains:

`makryvelios_dashboard_v2-8/app.py`

## New acceptance markers

- Header shows **POSTDOCTORAL ANALYTICAL ENGINE v5.8.1**.
- Module **12D. Agentic Research Mode** shows three intelligence engines: Smart offline semantic agent, Local AI — Ollama, External AI — user API key.
- Research conversation is a persistent multi-turn chat rather than separate generic offline/API answer boxes.
- Asking “What is the weakest finding and what can I not safely conclude?” returns a result grounded in actual computed tables when a model has been run.
- Naming an OLS term retrieves its exact coefficient/CI/p-value when available.
- Research-question generation supports data-aware offline mode and Local/API AI grounded generation.

No new mandatory dependency is introduced by this hotfix. Local Ollama is accessed over its HTTP endpoint and remains optional.
