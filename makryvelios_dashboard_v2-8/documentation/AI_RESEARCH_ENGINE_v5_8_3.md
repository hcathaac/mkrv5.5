# AI / LLM Research Engine v5.8.3

## Research-question reliability

For Google Gemini, Agentic question generation now requests JSON that conforms to an explicit response schema. This avoids relying only on a textual instruction to “return JSON”.

The parser remains provider-independent and accepts strict JSON arrays, objects containing `questions`/`research_questions`, fenced JSON, embedded JSON, JSONL and numbered/bulleted question lists. If a batch is still malformed, the agent performs a compact repair retry. If the provider remains unusable, the remaining slots are filled by the deterministic data-aware generator so the research workflow continues. The UI reports the number of AI-grounded and deterministic-recovery questions.

## Connectivity test

After configuring a provider in the persistent sidebar, use **TEST AI CONNECTION**. The test sends a minimal prompt and confirms that the provider/model route is reachable. This separates key/model/network failures from later structured-output parsing issues.

## Evidence and numerical integrity

All numerical analysis, econometrics, optimisation and Monte Carlo outputs remain deterministic application results. The AI layer may plan, interpret, synthesize and formulate research questions but does not alter computed tables.
