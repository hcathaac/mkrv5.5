# Deploy v5.7.2 over v5.7.1

This is an additive hotfix/feature release. Keep the same repository, branch and Streamlit entry point. Replace files at their existing relative paths.

Minimum update files:

- `app.py`
- `mapping.py`
- `gams_ui.py`
- `gams_compat.py`
- `VERSION.txt`
- `README.md`
- `CHANGELOG.md`
- `DEPLOY_v5_7_2.md`
- `VALIDATION_v5_7_2.md`
- updated GAMS documentation/tests

The bundled boundary files in `data/greece_nuts2_2024.geojson` and `data/greece_nuts3_2024.geojson` are retained and are required for fully offline/no-key detailed maps.

Acceptance check: solve `Vangelis – SYN2 540` using original supplied GAMS inputs, then open **Maps & spatial**. The map must render without any map API key and export PNG/SVG/PDF/HTML. Open **GAMS diagnostics** to confirm model/solver/equation/X.l outputs.
