# Deploy v5.5.3 over the existing Streamlit app

The public Streamlit URL remains unchanged because the existing app, repository, branch and main-file path are retained.

## Minimum replacement

Inside the existing GitHub folder `makryvelios_dashboard_v2`, replace these files:

- `app.py`
- `analytics_core.py`
- `mapping.py`
- `VERSION.txt`

Commit the replacements to the same `main` branch. Do not create a second `makryvelios_dashboard_v2` folder inside the first one.

The Streamlit main-file path must remain:

`makryvelios_dashboard_v2/app.py`

Reboot the existing Streamlit app after the commit. A successful deployment displays `POSTDOCTORAL ANALYTICAL ENGINE v5.5.3`. The **Dataset relationship** selector contains four options, documentation sheets are excluded from combination defaults, uploaded filenames appear in dark text, and the GIS map loads without a CARTO API key.
