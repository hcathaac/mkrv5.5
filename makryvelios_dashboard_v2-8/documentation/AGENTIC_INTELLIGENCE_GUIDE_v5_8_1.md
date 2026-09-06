# Agentic Research Intelligence — v5.8.1

v5.8.1 keeps the complete v5.8.0 platform and upgrades only the intelligence/orchestration layer of **12D Agentic Research Mode**.

## Intelligence engines

### Smart offline semantic agent

Requires no AI model and no API key. The agent classifies the meaning of a research question, searches the actual computed result tables and uploaded-PDF evidence, and answers with the most relevant numerical/source rows. Dedicated routes exist for strongest finding, weakest finding, named model terms, conclusions, causality, literature, research questions and next analyses. Unmatched questions use semantic retrieval rather than a generic fallback paragraph.

### Local AI — Ollama

The user may connect a local/self-hosted Ollama endpoint and choose an installed model. No commercial API key is required. The model receives only a bounded evidence context assembled from the computed run and uploaded literature, plus recent conversation turns. It does not perform the numerical calculation itself and cannot silently change the approved analytical specification.

When the Streamlit application itself is running locally, the normal endpoint is `http://127.0.0.1:11434`. A Community Cloud deployment cannot directly access the end user's laptop localhost; use a reachable/self-hosted endpoint for that deployment.

### External AI — user API key

The retained external LLM bridge remains available. It receives retrieved computed/source evidence only after an explicit action.

## Multi-turn research conversation

The v5.8.1 Agentic tab stores recent turns for the active session. Questions can refer to earlier discussion. The same conversation surface is used by all three intelligence engines so switching engines does not create separate research-chair-style workflows.

## Research-question generation

Offline generation is now data-aware: it ranks observed correlation patterns, variable completeness/dispersion, usable group structures, detected time/geography dimensions and uploaded-literature terms before composing questions. Local or external AI can optionally generate question batches grounded in the actual schema, observed relationship leads and PDF page evidence, with deduplication and method tagging.

## Scientific boundary

AI is a planner/interpreter, never the source of numerical truth. Statistical/econometric/optimisation engines remain deterministic. Causal language requires an explicit causal design; PDF citations and quotations require final verification against the originals.
