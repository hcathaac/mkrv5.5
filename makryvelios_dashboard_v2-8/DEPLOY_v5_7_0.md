# Deploy v5.7.0 over v5.6.1

The repository, branch and Streamlit main-file path remain unchanged.

Keep the current application folder and main file:

`makryvelios_dashboard_v2-8/app.py`

Replace the existing v5.6.1 files with the contents of this package at the same relative paths. Do not create an additional nested application folder.

New v5.7.0 files include:

- `gams_compat.py`
- `gams_ui.py`
- `llm_bridge.py`
- `reference_gams/vangelis/*.gms`
- `documentation/GAMS_COMPATIBLE_ITA_STUDIO_v5_7_0.md`
- `documentation/LLM_COPILOT_v5_7_0.md`

A successful deployment shows `POSTDOCTORAL ANALYTICAL ENGINE v5.7.0`, sidebar module `12A.1B GAMS-compatible ITA Studio`, and the visible `LLM CO-PILOT · USER API KEY` panel.

All v5.6.1 and v5.5.3 modules must remain visible.
