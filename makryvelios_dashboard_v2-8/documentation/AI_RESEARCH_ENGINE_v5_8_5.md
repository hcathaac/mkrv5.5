# AI Research Engine v5.8.5 — Context-budget controls

The Agentic RQ generator now sends only bounded relevant context to hosted/local models. **Compact** is intended for free-tier providers: selected research-role variables first, bounded schema, top observed correlation leads and relevance-ranked PDF passages. **Standard** and **Extended** deliberately send more context.

Before generation, the UI displays a conservative input-token estimate, the configured maximum output tokens and their approximate total. Groq users receive an explicit warning near the 8,000 TPM ceiling observed for the selected free-tier model. The application never changes provider/model automatically.

The same context-profile choice is applied to full-draft synthesis. Numerical tables remain deterministic and unchanged.
