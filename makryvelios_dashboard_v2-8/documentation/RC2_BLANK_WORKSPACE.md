# v6.0.0-rc2 — blank workspace

Deploy from the existing new-branch, main file makryvelios_dashboard_v2-8/app.py. Upload only the inner makryvelios_dashboard_v2-8 folder of the upload ZIP at repository root, preserving paths. Commit to new-branch.

The following old files/directories are deleted in the full clean repository but GitHub upload cannot delete existing files. Remove them from new-branch separately (not main):

- makryvelios_dashboard_v2-8/research_questions.csv
- makryvelios_dashboard_v2-8/research_hypotheses.csv
- makryvelios_dashboard_v2-8/source_evidence_catalogue.csv
- makryvelios_dashboard_v2-8/reference_gams/
- makryvelios_dashboard_v2-8/reference_stata/
- makryvelios_dashboard_v2-8/archive/

Keep data/greece_nuts2_2024.geojson, data/greece_nuts3_2024.geojson and data/greece_region_lookup.csv. They are geographic reference infrastructure, not study observations. Keep .streamlit/config.toml.

Generic prompt templates and methodological defaults are retained, but historic research questions, case-specific budgets, project scores and weight matrices are not loaded. Historical technical documentation is retained for method provenance and does not populate analyses.

Validation: 13 focused tests passed including blank-workspace checks, unchanged geographic map assets, neutral map, connector and prior preservation checks. Six additional selected GAMS/ITA/core tests passed. Python compile checks passed. Streamlit UI tests not run here (dependency missing); verify live deployment and uploaded-data workflows before merging. GitHub history is not rewritten.
