# Deploy v5.8.0 over v5.7.2

Version 5.8.0 is additive. Replace files at the existing relative paths inside the deployed `makryvelios_dashboard_v2-8` directory; do not create a nested release folder.

Keep the existing Streamlit main-file path unchanged: `makryvelios_dashboard_v2-8/app.py`.

New top-level modules after rebuild:

- `12C. Frontier methods laboratory`
- `12D. Agentic Research Mode`

New source files include `frontier_methods.py`, `frontier_ui.py`, `agentic_research.py`, `agentic_ui.py` and `documentation/FRONTIER_AGENTIC_GUIDE_v5_8_0.md`.

v5.8.0 also adds DuckDB, PyArrow and SHAP to `requirements.txt` and extends data intake to Parquet/Feather/Arrow IPC. Reboot the Streamlit app after the GitHub commit so dependencies are installed.
