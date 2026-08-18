# Deployment and Operations Guide - Version 5.4.0

## Repository layout

Keep the complete `makryvelios_dashboard_v2` folder intact. Both `mcda.py` and `research_chair.py` are mandatory. The application also expects its catalogue CSVs and, for offline mapping, the files under `data/`.

## Streamlit Community Cloud

1. Commit the release files to GitHub.
2. Open Streamlit Community Cloud and create or manage the application.
3. Select the repository and branch.
4. Set the main path to `makryvelios_dashboard_v2/app.py` when the folder is below the root.
5. Deploy or reboot.
6. Confirm the header shows v5.4.0 and the sidebar shows nineteen modules, including 12B Research Command Chair and its Prompt Library tab.

## Post-deployment acceptance checks

- The upload control is visible and supports multiple files.
- The bundled R&D workbook loads when no file is uploaded.
- OLS, Monte Carlo, clustering, panel and MCDA modules run.
- Research Command Chair accepts PDFs, applies year/row/variable filters and produces a non-empty paper bundle.
- Greece maps load from bundled boundaries.
- Excel, HTML and publication ZIP downloads are non-empty.
- Dark-panel text is white or light purple and notification text is white.

## Confidentiality

Remove confidential workbooks before committing to a public repository. Runtime uploads are preferable for restricted data. A private repository or private host is required where organisational policy prohibits public source hosting.

## Recovery

Retain the prior release ZIP. If a deployment fails, inspect the build log first. Reverting should use a known release commit or prior ZIP; do not remove individual modules ad hoc because imports and navigation are coordinated across files.
