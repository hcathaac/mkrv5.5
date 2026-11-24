# Makryvelios Workbench 6.0.0-rc1 — integrated release candidate

Base: repository archive main commit 4a5a518dc8ea9bf0190126f3f397cf9ff497fe7e. This package already contains the applied upgrade. No installer is necessary. The main GitHub branch has not been changed.

## Deploy on a test branch

Unzip the full repository archive. For an existing GitHub repository use the separate upload-only ZIP, whose paths are relative to the repository root. Create a new branch from main, upload the extracted changed files (not the ZIP), retaining paths, and explicitly delete `makryvelios_dashboard_v2-8/data/rd_projects_reference.xlsx` on that branch. The ZIP cannot delete an existing remote file. Keep both `.streamlit/config.toml` files in the appropriate directories (repository root and app directory). Streamlit Cloud entry point remains `makryvelios_dashboard_v2-8/app.py`. On Streamlit Cloud create a separate test app pointing to the new branch; use the same Python version as the existing working deployment. Do not merge until UI, uploads, old modules, reset, live API and MCP tests pass.

The historical workbook remains in Git history. The original SYN2 GAMS reference inputs, geographical data and method catalogues are retained; the GAMS Studio defaults to the active user dataset, and original GAMS examples can be explicitly selected. Review historical public data for confidentiality.

## Coverage and limitations

| Request | Candidate implementation |
|---|---|
| Blank start | No auto-loaded workbook; independent tools remain accessible; old R&D functions retained. |
| Cache / restart | Clear computed session state and two parser caches; full session reset clears uploads and API keys. Parser caches are process-wide; clearing may affect other users' cached performance. This is not an OS/server restart. |
| Cleaning | Per-source audit, reversible preview and application to active analysis inputs, duplicate and missing-value codes, explicit numeric conversion, hashes and change log. Existing intake tidying precedes this layer; it is not raw-byte archival. |
| Public loans | Regular period aggregation (flows vs stocks), naive and linear-trend baselines, chronological backtesting, MAE/RMSE, empirical one-step error bands and illustrative fixed-rate debt-service schedules. No sovereign risk, causal impact, eligibility or loan approval inference; no calibrated multi-step uncertainty. |
| Conferences | Live Brave Search API lookup with user key and browser-search link without key. Results are unverified leads; no embedded general web browser or automatic official deadline verification. |
| UX | Streamlit dark theme and CSS for tables, controls, chat, dropdowns, hover/focus; requires visual/browser regression checks. |
| Methods / Chair | Family filter, original menu retained, explanations and PDF/DOCX ingestion in Chair and Agentic mode. DOCX block numbers are not physical pages. Legacy .doc is unsupported. |
| Outputs | Existing 600-dpi/vector export pathways retained; new forecast exports in colour and grayscale, PNG/SVG/PDF; stronger external Chair instruction and traceable table digest in drafts. Not a guarantee of a fully written or publishable paper; bounded previews and human scientific verification are required. |
| Zotero | Read-only live top-level items (max 500) or CSL/Zotero JSON import, DOI/title duplicate flags, metadata audit, source JSON and report ZIP. No write-back, attachment sync, full-text review or formatted bibliography styles. |
| Overview | Greek/English keyword router, independent source audits, method catalogue and module navigation. Explicit variable mapping still needed. |
| LLM connector | Optional local MCP stdio adapter uses actual existing analytical engine, audits all supplied datasets, then runs one explicitly mapped source through bounded Research Chair commands. No automatic joins, remote ChatGPT registration, full coverage of every UI tool, or authentication for remote deployment. |

## Optional MCP adapter (local only)

Install `mcp` separately in the app virtual environment. Point `MKRV_DATA_DIR` at a local folder containing only the datasets and documents explicitly authorised for this client. Example client configuration (adapt paths):

```json
{"mcpServers":{"makryvelios":{"command":"/absolute/path/.venv/bin/python","args":["/absolute/path/workbench_connector.py"],"env":{"MKRV_DATA_DIR":"/absolute/path/authorised-research-data"}}}}
```

The local MCP tool `research` accepts `question`, `datasets` (filenames), `active` (returned source ID), `roles` (e.g. outcome, predictors, year_column) and optional `documents`. First call with no active source to see audits and available mappings; the second call can execute the requested route. Returned tables are capped at 200 rows, with total row counts. No arbitrary shell, Python evaluation or arbitrary filesystem path is exposed. Verify that the chosen question is supported by the existing Chair router and inspect its feasibility verdict. No automatic data transmission to GitHub is needed or intended.

ChatGPT or other remote LLM interfaces require a separately deployed authenticated remote MCP service, per-client connection configuration, user-level access control, encrypted transport, upload policy and integration tests. A GitHub connector alone is not a runtime and cannot execute Streamlit merely by reading source. Do not expose this stdio adapter directly as an unauthenticated network service.

## Verification performed in this environment

- Applied installer anchors to the actual supplied repository; all Python files compile.
- 9 targeted tests passed (7 new tests including DOCX evidence and three-source connector execution, plus 2 preservation tests). The connector test exposed and corrected an inherited CSV parser bug involving low_memory with the Python engine.
- Full legacy unit suite was attempted but exceeded the execution limit after 50 passing tests and one confirmed dependency failure (linearmodels missing). It was not completed. The panel test failed because linearmodels could not be installed; it remains in requirements. No numerical panel regression validation is claimed.
- Streamlit, linearmodels, xlrd, pyarrow, duckdb, and MCP SDK were unavailable locally; pip installation failed. Streamlit AppTest, live Brave/Zotero calls, MCP transport, browser styling and production deployment remain unverified. The existing smoke script now checks v6 and blank startup; the original interaction script uses a synthetic fixture in a temporary test script instead of the removed workbook. These UI tests still require actual execution.
- Local connector executes bounded Research Chair commands, not every UI feature. The remote ChatGPT connector is not registered or hosted. No full embedded browser or complete bibliography citation-style manager is supplied.

## Run tests in a configured environment

```sh
cd makryvelios_dashboard_v2-8
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python tests/app_smoke.py
python tests/app_interactions.py
streamlit run app.py
```

Optional MCP: install `mcp` separately and use the configuration above. The UI cache reset is not a server/OS restart. Historical outputs and publication figure routes remain available, subject to the actual input requirements and regression verification.
