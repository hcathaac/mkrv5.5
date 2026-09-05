# v5.8.0 Validation Record

## Preservation

- Source base: complete v5.7.2 repository package.
- File-path preservation comparison: **0 v5.7.2 files missing** after the upgrade.
- Critical retained engines (`ita.py`, `ita_ui.py`, `gams_compat.py`, `gams_ui.py`, `respondent.py`, `respondent_ui.py`, `mapping.py`, `research_chair.py`, `mcda.py`, `advanced_analytics.py`, `visuals.py`, `reporting.py`, `llm_bridge.py`) remained byte-for-byte unchanged.

## Executed automated tests in the build sandbox

- New v5.8.0 Frontier/Agentic + preservation + retained GAMS-map tests: **9 passed**.
- Retained v5.7.2 core tests executed in bounded groups: **35 passed**.
- Total executed passing tests: **44**.
- The new tests include actual synthetic execution of Pareto/HiGHS MILP, cross-fitted AIPW, Bayesian posterior/predictive simulation, SHAP/fallback explainability, 150-question generation, offline Agentic analysis, PDF source indexing and construction/opening of the complete submission ZIP.

## Environment-limited checks

- The pre-existing panel-model test requires `linearmodels`. The build sandbox does not contain that dependency, so that single retained test cannot execute here. `requirements.txt` already retains `linearmodels>=6,<8` for the Streamlit deployment.
- Streamlit AppTest cannot execute in this sandbox because Streamlit itself is not installed. The AppTest source was updated to assert both new v5.8.0 modules and retains the full module-render loop for deployment/runtime validation.
- DuckDB and PyArrow are newly declared v5.8.0 deployment dependencies but are not installed in this offline build sandbox. Their UI fails safely with an installation/reboot message; the corresponding runtime paths activate after Community Cloud installs the updated requirements.

## Compile and package checks

- All production/test Python source files are compiled before packaging.
- Full, replace-only and hotfix ZIPs are opened with the Python ZIP library after creation to verify archive integrity.
- The Agentic package test verifies DOCX, XLSX, JSON, HTML and 600-dpi/vector figure members.
