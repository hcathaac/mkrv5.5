# v5.8.2 update note

After uploading the v5.8.2 hotfix, reboot Streamlit. Confirm the sidebar shows the expanded **AI / LLM RESEARCH ENGINE** provider selector and that 12D Agentic Research Mode exposes automatic AI synthesis plus the full-draft refinement button. Existing main-file path and deployment structure remain unchanged.

# v5.8.1 Agentic intelligence hotfix

Upload the v5.8.1 hotfix contents into the existing `makryvelios_dashboard_v2-8` folder and commit to `main`. The Streamlit entrypoint remains unchanged. Acceptance marker: header v5.8.1 and three Agent intelligence engines inside 12D.

# Update the GitHub and Streamlit app — v5.8.0

The deployed GitHub folder is `makryvelios_dashboard_v2-8` and the Streamlit main-file path remains `makryvelios_dashboard_v2-8/app.py`.

## Browser-only update

1. Download and extract the v5.8.0 replace-only or hotfix ZIP.
2. Open the GitHub repository and enter the existing `makryvelios_dashboard_v2-8` folder.
3. Choose **Add file → Upload files**.
4. Upload the *contents* of the update ZIP into that folder so matching files are replaced; do not create a nested release folder.
5. Commit directly to `main` unless a separate branch is intentionally being used.
6. Streamlit Community Cloud normally rebuilds automatically. If not, open **Manage app → Reboot app**.
7. Keep the main file path `makryvelios_dashboard_v2-8/app.py`.

## v5.8.0 acceptance markers

- Header: **POSTDOCTORAL ANALYTICAL ENGINE v5.8.0**.
- Existing modules 1 through 13 remain present.
- New **12C. Frontier methods laboratory**.
- New **12D. Agentic Research Mode**.
- Agentic mode can generate up to 150 research questions in one batch and run offline without an API key.
- The sidebar LLM key remains optional.
- Data intake accepts the previous Excel/CSV/TSV formats plus Parquet/Feather/Arrow after the new dependencies install.

## Dependency note

The v5.8.0 `requirements.txt` adds DuckDB, PyArrow and SHAP. Reboot after the GitHub commit so Community Cloud installs the updated requirements.
