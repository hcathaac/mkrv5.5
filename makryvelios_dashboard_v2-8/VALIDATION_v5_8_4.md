# Validation v5.8.4

Acceptance checks:

- Groq `openai/gpt-oss-120b` receives `response_format.type = json_schema` and `strict = true` when Agentic full-draft synthesis is requested.
- No automatic provider/model fallback occurs on an API error.
- Strict JSON synthesis parses successfully.
- Markdown/plain-text sectioned synthesis with Abstract, Results, Discussion, Conclusion and Limitations is recovered from the same response.
- Deterministic numerical tables are not modified by AI synthesis.
- Existing v5.8.3 RQ reliability and prior GAMS/ITA/Frontier functions are retained.
