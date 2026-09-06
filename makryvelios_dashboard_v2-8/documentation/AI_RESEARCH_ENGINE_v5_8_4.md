# AI / LLM Research Engine v5.8.4

The Agentic full-draft synthesis uses explicit structured-output contracts where supported. For Groq GPT-OSS models, the application sends a strict JSON Schema through the OpenAI-compatible `response_format` field. This makes Abstract, Results, Discussion, Conclusion, Limitations and key findings machine-readable without changing the selected provider/model.

If the selected model nevertheless returns a complete sectioned prose answer, the same response can be recovered into the manuscript sections. This is parsing of the selected model response, not provider/model fallback. Provider errors remain visible to the user.

Numerical analysis remains deterministic and is never recomputed by the LLM.
