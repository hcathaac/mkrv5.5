# Makryvelios Research Analytics & Econometrics Command Centre v5.8.0

This is a complete replacement package for the existing Streamlit app. It retains the original R&D data audit, variable dictionary, project/regional modelling, region-year panel, Greece GIS, spatial diagnostics and scenario functions, and adds schema-agnostic multi-file analysis.

Version 5.8.0 is a strictly additive frontier/automation upgrade over v5.7.2. It adds a separate **12C Frontier methods laboratory** (Pareto/robust optimisation, cross-fitted AIPW causal inference, offline Bayesian posterior modelling, SHAP explainability and DuckDB/Arrow/Parquet scaling) plus a standalone **12D Agentic Research Mode**. Agentic mode works without an AI API, can index literature PDFs locally, generate up to 150 research questions per batch, build an approval-gated research plan, execute bounded analytical routines, draft evidence-grounded discussion/conclusions and assemble a near-submission package containing DOCX, XLSX, JSON, interactive HTML and publication graphics. A user-key LLM can optionally refine synthesis after computation; it is never required for the statistical/optimisation engines. No earlier feature is removed.


Version 5.7.2 is a strictly additive visual/diagnostic upgrade over v5.7.1. The GAMS-compatible ITA Studio now includes API-key-free Greece maps rendered entirely from bundled Eurostat GISCO NUTS-2/NUTS-3 GeoJSON, fine NUTS-3 border/coastline overlays, GREEN/GRAY/RED ITA maps, allocation/utilisation/project-exposure/score-contribution maps, optional Monte Carlo stability maps, exploratory Moran diagnostics, and 600-dpi PNG plus vector SVG/PDF/interactive HTML export. It also exposes GAMS-style model statistics, HiGHS MIP diagnostics, equation listings, X.l variable listings, scenario-overlap tables, solver messages and .lst-style reproducibility files. No existing function is removed.

Version 5.7.1 is a strictly additive maintenance upgrade over v5.7.0. It fixes regional budget auto-mapping, embeds the supplied SYN2 540 raw GAMS input tables for one-click exact replication, and strengthens dark-theme caption contrast. No capability is removed.

Version 5.7.0 is a strictly additive upgrade over v5.6.1. It adds a separate **12A.1B GAMS-compatible ITA Studio** that keeps Evangelos Makryvelios' GAMS algebraic logic visible while executing the equivalent binary MIP with SciPy/HiGHS. It includes the original SYN2 540 and R&D 2437 presets, regional/sector/intervention constraints, GREEN/RED `X.fx` logic, effective-budget rules, Monte Carlo, original `.gms` source viewing, `.gms/.prn` exports and reproducibility packages.

A visible sidebar **LLM Co-pilot · User API Key** panel is also added. The key is session-only. External LLM use is optional and explicit; numerical analysis and optimisation never depend on it. The Research Command Chair and the GAMS Studio can use the configured model for interpretation and drafting after deterministic computation.

Version 5.6.1 retains the complete v5.6.0 ITA upgrade and adds a separate **12A.2 Expert respondent analytics** module. It analyses complete respondent-level preference distributions, agreement, dependence, latent structure, exploratory preference segments and subgroup heterogeneity, then passes the validated empirical weight vectors directly to Hybrid ITA-RW. Every v5.5.3 capability remains available.

The respondent module is schema-agnostic and therefore operational before the two papers arrive. It does not assume their findings: the exact respondent identifier, C1-Cn fields and optional subgroup variables are mapped at run time. The resulting empirical distribution can be activated for Hybrid ITA-RW immediately, while the papers' final confirmatory models and interpretations can later be reproduced without rebuilding the interface.

## Documentation library

Version 5.8.0 adds `FRONTIER_AGENTIC_GUIDE_v5_8_0.md` covering Pareto/robust optimisation, causal AIPW, Bayesian posterior analysis, SHAP explainability, DuckDB/Arrow/Parquet and offline/LLM-assisted Agentic Research Mode.

Version 5.6.1 adds `EXPERT_RESPONDENT_ANALYTICS_GUIDE_v5_6_1.md`, including the data contract, descriptive and inferential outputs, compositional caveats, preference segmentation and empirical ITA bridge.

Version 5.6.0 adds `ITA_DECISION_SUPPORT_GUIDE_v5_6_0.md`, covering data mapping, both ITA variants, colour semantics, round/scenario visualisation, maps/spatial diagnostics, GAMS replication and the planned expert-distribution connection.

Version 5.5.0 retains every v5.4.0 capability and adds a one-screen Guided Chat Autopilot, editable selected/all-question batches, explicit feasibility verdicts, genuine paper-ready prose answers, and Research Chair figures in standalone interactive HTML plus colour/black-and-white 600-dpi PNG and vector SVG/PDF.

Version 5.4.0 retained every v5.3.2 capability and added:

- an in-app library of 52 copy-ready questions and analytical commands covering all nineteen modules;
- deterministic Research Chair commands for Monte Carlo OLS with explicit seed/repetitions, safe-versus-prohibited conclusions, limitations, further-analysis priorities, research-question development, methodology/results outlines, PDF-method replication audits and equation/algorithm checks;
- a plain-language interpretation panel for every table and chart, explaining what it is, why it is used, how to read it, what the observed pattern means, what it cannot establish and what to do next;
- automatic `Output guide` and `Prompt library` worksheets in Research Chair workbooks, plus CSV/Markdown guides in research and publication bundles;
- beginner-accessible explanations without weakening the statistical safeguards or removing any analytical function.

Version 5.3.2 added:

- ranked answers to headline questions such as “What is the most striking statistical result?”;
- direct magnitude, sample-size and p-value reporting, scientific meaning, data-quality caveats, model-specification warnings, publication-ready wording and next-step advice;
- automatic suppression of identifier, code and time fields from headline-correlation ranking;
- an in-app library of example questions and the expected answer structure;
- black typed and selected text on every light input surface.

Version 5.3.1 added:

- non-empty all-numeric descriptive output when no model variables have been selected;
- variable-level missingness and method-evidence tables;
- deterministic natural-language commands that execute analysis on the saved XLSX/CSV scope before answering;
- black typed/user-message text on light controls and chat surfaces.

Version 5.3.0 added:

- `RESEARCH_COMMAND_CHAIR_GUIDE.md` — free/offline XLSX/PDF evidence selection, algorithms, safe equations, natural-language interpretation and paper-report workflow.

- `Makryvelios_Technical_Documentation_v5_2_1.docx` and `.pdf` — consolidated technical report and user manual.
- `COMPLETE_DOCUMENTATION.md` — searchable source version of the consolidated documentation.
- `QUICK_START.md` — installation and minimum safe analytical workflow.
- `MCDA_GUIDE.md` — complete operating and interpretation guide for the dedicated MCDA engine.
- `DEPLOYMENT_AND_OPERATIONS.md` — GitHub/Streamlit deployment, acceptance and recovery procedures.
- `VALIDATION_AND_QA.md` — automated and scientific validation protocol.
- `REQUIREMENTS_COVERAGE.md` — requirement-to-implementation coverage matrix.

The full documentation is also downloadable from **Module 13 — Methods & reproducibility** when the corresponding files are included in the deployed package.

## What is new

- A new **Research Command Chair** that accepts research questions, algorithms, equations, ordered steps, limitations and notes without a paid AI API.
- Exact selection of spreadsheet variables, rows and year ranges, plus up to six simultaneous row filters; the original dataset is never modified.
- Multiple-PDF upload with document, page-range and keyword selection, page-level evidence indexes and transparent safeguards for scanned PDFs and quotations.
- Safe derived-variable expressions using a restricted mathematical grammar; arbitrary code execution is blocked.
- Reproducible descriptive, longitudinal, correlation and HC3 OLS protocol execution, natural-language replies, equations and Word/Markdown paper blueprints.
- Optional local Ollama enhancement. The built-in offline interpreter and every statistical/export function remain available without Ollama, subscriptions or API keys.
- A complete Research Chair bundle containing the filtered dataset, selected PDF evidence, protocol JSON, result tables, XLSX workbook, paper blueprint, interpretation guide and full copy-ready prompt library.

- A complete dark postdoctoral command-centre interface with responsive glass/cyber styling and a redesigned, high-contrast multi-file upload control.
- A new **Advanced clustering & segmentation** laboratory supporting one-variable absorption clustering and multivariate K-means, hierarchical agglomerative clustering, Gaussian mixtures and DBSCAN.
- Automatic cluster-number selection with silhouette, Calinski–Harabasz and Davies–Bouldin diagnostics, cluster profiles, PCA/one-dimensional projection and downloadable publication bundles.
- A new **Panel model laboratory** with pooled OLS, two-way fixed effects, random effects, entity-clustered covariance and a Hausman FE–RE specification test.
- A new **Predictive model laboratory** comparing OLS, Ridge, Lasso, Elastic Net, random forest, extra trees and gradient boosting by honest out-of-fold metrics and permutation importance.
- Huber robust regression and Gamma log-link regression added to the single-outcome econometric laboratory.
- Hypothesis tests now report effect sizes, including Hedges g, rank-biserial correlation, eta-squared, epsilon-squared and Cramér's V visualisation.
- Explicit HTML/data downloads and interpretive panels expanded across descriptive, hypothesis, panel, time-series, PCA, scenario and allocation figures.

- A visibly labelled **OLS Studio**: OLS is the default estimator and now has step-by-step guidance, downloadable fit/coefficient/diagnostic tables, observed-versus-fitted and residual plots, a coefficient forest plot, and a colour/black-and-white publication bundle.
- A dedicated **Monte Carlo & uncertainty laboratory** with wild bootstrap, residual bootstrap and parametric-normal OLS simulation. It exports every draw, empirical bias, Monte Carlo standard error, sign probability and percentile confidence interval.
- A stochastic **R&D portfolio selection** tool, grounded in the supplied Makryvelios research context, that propagates cost/benefit uncertainty and reports project selection probabilities plus downside portfolio distributions.
- A high-technology visual redesign, an analysis navigator, module-level operating instructions, interpretation warnings and more explicit table/figure downloads.

- Simultaneous upload of multiple `.xlsx`, `.xls`, `.xlsm`, `.csv`, `.tsv`, `.parquet`, `.pq`, `.feather` and `.arrow` files.
- Reads one or every Excel sheet. Files may be kept separate, appended by column name, or joined on one or more keys.
- Automatic repair of the supplied R&D workbook's two-row header (the `1–83` index row followed by the real variable names).
- Up to 1,000 dependent and 1,000 independent variables can be selected. A vectorised SVD engine performs large multi-outcome OLS screens; detailed models then use robust diagnostics.
- Descriptive statistics, categorical frequencies, correlations and p-values, group tests, categorical association tests and normality checks.
- OLS, WLS, logit, probit, Poisson, negative-binomial GLM, fractional logit and quantile regression; HC0–HC3, HAC and clustered covariance; categorical/fixed effects; VIF and extensive residual diagnostics.
- Original EE1–EE9 R&D project and region-year specifications recovered from the supplied Stata research-question document and `.do` files.
- Official Eurostat GISCO 1:1 million NUTS-2 and NUTS-3 boundaries, bilingual Greek-region matching, interactive colour/monochrome maps, Moran's I and local cluster diagnostics.
- Publication packages containing colour and black-and-white versions in 600-dpi PNG plus vector SVG and PDF, together with the plotted data and notes.
- A dedicated MCDA decision laboratory with MAVT, TOPSIS and PROMETHEE II; equal, user-defined, Entropy, CRITIC and AHP pairwise weighting; AHP consistency diagnostics; weight sensitivity; Monte Carlo rank acceptability; method-agreement tables; and complete publication bundles.
- Self-contained HTML/JavaScript analytical reports.
- PCA, standardised k-means, ADF/KPSS time-series diagnostics, econometric shock simulations and constrained resource-allocation optimisation.
- Optional independent R replication script (`r_engine.R`). Python operation does not depend on R.

## Run it locally

Python 3.11 or 3.12 is recommended.

```bash
cd makryvelios_dashboard_v2
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser normally opens at `http://localhost:8501`. If it does not, copy the URL printed in the terminal.

## Use the data

The package may include the original 3,259-row R&D workbook in `data/`. It loads automatically when the app starts. As soon as files are uploaded through the sidebar, the uploads become the active sources for that browser session.

For the Antonis Tritsis dataset, upload the workbook or CSV and select the sheet containing the project-level table. Choose:

1. **Keep datasets separate** to analyse one workbook/sheet at a time.
2. **Combine columns side-by-side (by row order)** when files contain the same ordered observations but different variables. All sources are selected by default and every resulting column becomes available to the menus.
3. **Append rows** when files have the same or overlapping columns.
4. **Join datasets on key(s)** when R&D and Antonis Tritsis tables or external denominators must be linked by a stable key such as NUTS code, region and/or year.

Do not join merely on region names if rows have a finer grain. Verify grain and uniqueness first in **Data hub & audit**.

## Deploy on Streamlit Community Cloud

1. Extract the ZIP.
2. Create a GitHub repository and upload the contents of `makryvelios_dashboard_v2-8` to the repository root.
3. Commit `app.py`, all `.py` modules (including `mcda.py`), `requirements.txt`, `.streamlit/config.toml`, the two research catalogue CSVs, and (only if appropriate) the `data/` workbook.
4. In Streamlit Community Cloud, choose **Create app** and select that repository.
5. Set the main file path to `app.py` and deploy.

The Greece map fetches public boundaries from Eurostat GISCO. If the host blocks that request, upload the required GeoJSON in the GIS module. Do not commit confidential data to a public repository; use a private repository or upload the data at run time.

## GitHub Pages is not the host

GitHub Pages cannot execute Python/Streamlit. GitHub stores the source; Streamlit Community Cloud (or another Python host) runs it. The portable HTML reports downloaded from the app can be placed on ordinary static hosting, but they are result snapshots rather than the live analytical application.

## Statistical limits and honest interpretation

The 1,000 × 1,000 selector is real, but statistical identification still depends on sample size, rank, missingness and theory. One thousand predictors cannot be uniquely estimated from fewer than roughly one thousand independent observations without regularisation or dimensionality reduction. The wide engine therefore uses a pseudoinverse and is explicitly labelled as screening; shortlist models must be re-estimated in the detailed laboratory with robust or clustered inference.

P-values are adjusted using Benjamini–Hochberg or Bonferroni. Results remain sensitive to measurement, repeated regional values, multicollinearity, sparse outcomes and endogenous selection. Associations must not be described as causal without a defensible identification strategy.

## Optional R replication

Install R separately and then install the R packages needed by the selected estimator:

```r
install.packages(c("jsonlite", "sandwich", "lmtest", "MASS"))
```

`r_engine.R` accepts a CSV, a JSON model configuration and an output folder. It is provided for independent checking; the Streamlit app remains Python-first and fully usable without R.

## File map

- `app.py` — Streamlit interface and all modules.
- `analytics_core.py` — ingestion, statistics, hypothesis tests, econometrics, batch engine and exports.
- `advanced_analytics.py` — validated clustering, panel-model comparison and cross-validated predictive analytics.
- `mcda.py` — dedicated multi-criteria ranking, weighting, robustness analysis and publication bundles.
- `ita.py` — ITA-PB and Hybrid ITA-RW engines, exact MILP optimisation, uncertainty rounds, scorecards and GAMS-ready reproducibility exports.
- `ita_ui.py` — self-contained ITA mapping, scenario, visual decision-support, spatial and export interface.
- `respondent.py` — respondent-level preference statistics, consensus, PCA, segmentation, subgroup testing and reproducibility export.
- `respondent_ui.py` — expert-data mapping, visual analytics and empirical Hybrid ITA-RW bridge.
- `legacy_rd.py` — compatibility with the original 83-variable R&D workbook and EE1–EE9 region-year panel.
- `mapping.py` — Greece GIS, official boundaries, Moran/LISA diagnostics and static map exports.
- `visuals.py` — interactive and publication-quality colour/black-and-white figures.
- `reporting.py` — portable self-contained HTML reports.
- `research_chair.py` — offline PDF/data scoping, safe equations, research protocols, natural-language interpretation and paper-report bundles.
- `r_engine.R` — optional R replication.
- `research_questions.csv` — nine source R&D questions plus clearly labelled Antonis Tritsis extensions.
- `research_hypotheses.csv` — the nineteen recovered R&D hypotheses.
- `source_evidence_catalogue.csv` — traceable links between supplied articles, documented methods and app modules.
- `documentation/` — complete user, technical, MCDA, deployment, validation and requirements documentation.
- `tests/test_core.py` — deterministic smoke/unit tests.
- `tests/app_smoke.py` — render coverage for all nineteen modules.
- `tests/app_interactions.py` — interaction coverage for OLS, Monte Carlo, clustering, prediction, panel and MCDA paths.

## Recommended workflow

1. Load and audit sources.
2. Select the exact research question.
3. Confirm unit of analysis and construct the project or region-year table.
4. Run descriptive and missingness checks.
5. Estimate a theory-led primary model.
6. Inspect diagnostics and multiplicity.
7. Run robustness estimators and spatial checks where appropriate.
8. Export exact tables and both colour and black-and-white publication figures.
9. Record the dataset version, filters, model configuration and caveats in the HTML report.

## Updating the GitHub/Streamlit deployment

This package targets the existing deployed folder `makryvelios_dashboard_v2-8`; keep that folder and its Streamlit entrypoint path unchanged. Replace the files inside the actual deployed repository folder and retain its existing Streamlit entrypoint path. Streamlit rebuilds automatically after the GitHub commit.
