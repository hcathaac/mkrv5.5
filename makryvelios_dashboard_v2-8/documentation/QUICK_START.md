# Quick Start - Version 5.4.0

Version 5.4.0 retains all prior workflows and adds 52 copy-ready prompts plus a plain-language guide for every exported table and chart. Use Module 12B after data audit to restrict variables, rows, years and PDF evidence, document the research protocol, select a prompt from tab 5, and export the results, interpretation guide and paper blueprint.

## First run

```bash
cd makryvelios_dashboard_v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Minimum safe workflow

1. Upload one or more Excel/CSV files in the sidebar.
2. Choose whether files remain separate, are appended or are joined on verified keys.
3. Open **Data hub & audit** and verify rows, variables, types, duplicates, missingness and outliers.
4. Open **Research questions** and select the exact analytical objective and unit of analysis.
5. Run descriptive analysis before modelling.
6. Select the matching econometric, spatial, clustering, predictive, panel or MCDA module.
7. Read assumptions, diagnostics and explanatory comments.
8. Download the exact result tables and publication bundle.
9. Archive the input file, software version, model configuration and seed.

## Copy-ready questions

Open **12B. Research Command Chair → 5 · Copy-ready Prompt Library**. Filter by topic or dashboard module, copy the prompt, replace square-bracket placeholders, complete the stated setup, and paste it into the question box. Use specialist modules when the prompt names one: the Research Chair explains and scopes the request, while the named module performs its validated estimator.

Every table and chart now displays a short explanation and a detailed **How to read this output** panel. Every workbook includes an `Output guide` sheet; publication bundles include the same guidance in CSV and Markdown.

## MCDA quick start

1. Open **12A. Dedicated MCDA engine**.
2. Select the project/region identifier and at least two numeric criteria.
3. Set each criterion to Maximise or Minimise.
4. Select Equal, User-defined, Entropy, CRITIC or AHP weights.
5. Select MAVT, TOPSIS and/or PROMETHEE II.
6. Retain a reproducible seed and configure sensitivity/Monte Carlo draws.
7. Run the engine and compare rankings, correlations and rank acceptability.
8. Download the complete workbook and publication bundle.

MCDA ranks are conditional on the stated preference model and must not be described as causal effects or probabilities of project success.

## Streamlit deployment

Upload the complete folder to GitHub, then deploy from Streamlit Community Cloud. If the folder is below the repository root, use `makryvelios_dashboard_v2/app.py` as the main file. GitHub itself does not display the Streamlit deployment control.
