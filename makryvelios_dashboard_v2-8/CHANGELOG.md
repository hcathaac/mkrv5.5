# v5.8.5 — 2026-09-06 — Compact visible AI context controls

- Additive hotfix only; no feature/module/export removal.
- Fixed Groq/free-tier RQ failures caused by oversized input context: the Agentic RQ prompt no longer serialises the full 83+ variable schema and dozens of PDF passages.
- Added bounded context profiles: **Compact — free-tier friendly**, **Standard**, and **Extended**.
- Added explicit user-focus routing from mapped outcome/predictors/group/time/geography fields into AI research-question generation.
- Added relevance-ranked PDF evidence retrieval and bounded observed-correlation leads for RQ generation.
- Added a visible **AI questions per request** control; large RQ banks are split into smaller requests without silently changing provider or model.
- Added a conservative visible input-token estimate and Groq 8,000-TPM warning/safe indicator before Generate.
- Applied the same manually selected context profile to full-draft AI synthesis so free-tier calls can be intentionally bounded.
- No automatic provider/model fallback was added. Deterministic calculations and all previous modules remain unchanged.

# v5.8.4 — Agentic structured-synthesis reliability

- Additive hotfix only; no feature removals.
- Groq GPT-OSS structured synthesis now sends `response_format: json_schema` with `strict: true` when a synthesis schema is requested.
- Agentic full-draft rewrite explicitly requests the synthesis schema.
- The synthesis parser also recovers complete Abstract/Results/Discussion/Conclusion/Limitations from sectioned prose returned by the same model.
- No automatic provider/model fallback is introduced; provider/model selection remains manual and visible.
- Deterministic numerical tables remain unchanged.

# v5.8.3 — 2026-09-06

- Added Gemini schema-constrained structured output for Agentic research-question generation.
- Hardened the RQ parser for fenced/embedded JSON, wrapped objects, JSONL and numbered/plain-text question lists.
- Added automatic constrained retry for malformed model output.
- Added deterministic data-aware recovery so an AI formatting failure can no longer abort the full research-question batch.
- Added generation diagnostics showing AI-grounded vs recovered questions and parse-repair events.
- Added **TEST AI CONNECTION** in the persistent AI/LLM panel.
- No previous capability removed or renamed.

# v5.8.2 — 2026-09-06

- Strictly additive AI-synthesis hotfix over v5.8.1; no existing analytical, optimisation, mapping, export or Agentic capability removed.
- Expanded the persistent AI/LLM panel with Google Gemini (free tier available), Groq (free plan available), Ollama Local (no API key), Anthropic Claude and generic OpenAI-compatible endpoints.
- Added provider-native REST handling for Gemini and Ollama and a Groq OpenAI-compatible route, without adding a mandatory SDK dependency.
- Agentic Research Mode now defaults to the configured AI engine when one is available and can automatically run an evidence-grounded synthesis pass after deterministic computation.
- The AI synthesis pass rewrites Abstract, Results, Discussion, Conclusion and run-specific Limitations from retrieved computed rows and uploaded-PDF page evidence; deterministic numerical tables are never altered.
- Added a one-click "REFINE / REWRITE ENTIRE DRAFT WITH SELECTED AI" action and retained the original deterministic narrative for audit.
- The DOCX/HTML/Markdown submission outputs now use the AI-refined Results/Discussion/Conclusion sections when available and preserve the offline draft in the run state.
- Added explicit privacy notice for Gemini free-tier usage and no-key/local deployment guidance for Ollama.

# v5.8.1 — 2026-09-06

- Strictly additive Agentic intelligence hotfix over v5.8.0; no existing capability removed.
- Replaces generic keyword-style Agentic chat with evidence-aware semantic routing over actual computed result rows and PDF page evidence.
- Adds specific strongest/weakest/model-term/causality/literature routing and semantic retrieval fallback.
- Adds a persistent multi-turn research chat with quick evidence questions.
- Adds Local AI via configurable Ollama endpoint/model with no API key; deterministic analysis remains independent of AI.
- Retains external user-key LLM as an optional third intelligence engine.
- Upgrades the 150-RQ generator: offline questions are ranked from observed data patterns, while Local/API AI can generate grounded batches from schema + relationship leads + PDF evidence.
- Preserves approval gates, deterministic numerical engines and all v5.8.0/v5.7.x modules.

# v5.8.0 — 2026-09-06

- Strictly additive upgrade over v5.7.2; no previous module or export path removed.
- Added **12C Frontier methods laboratory** with Pareto/multi-objective/robust binary portfolio optimisation, cross-fitted doubly robust AIPW causal estimation, offline Bayesian posterior/predictive modelling, SHAP explainable ML and DuckDB/Arrow/Parquet large-data tooling.
- Added **12D Agentic Research Mode** as a separate approval-gated workflow that functions without an AI API.
- Agentic mode locally indexes literature PDFs, detects source identifiers, creates extractive evidence notes, generates up to 150 research questions per batch, builds a transparent plan, runs bounded validated analyses, drafts discussion/conclusions and exports a near-submission research package.
- Added DOCX + XLSX + JSON + HTML + 600-dpi PNG + vector SVG/PDF outputs to the Agentic submission package where generated.
- Connected optional user-key LLM synthesis to Agentic mode only after deterministic computation and explicit user action; numerical engines remain AI-independent.
- Extended the intake layer to Parquet, Feather and Arrow IPC while retaining every XLS/XLSX/XLSM/CSV/TSV workflow.
- Added read-only DuckDB SQL, compressed Parquet and Arrow IPC exports.
- Added explicit approval gates and scientific safeguards for causal identification, PDF citation verification and near-submission draft status.

# v5.7.2 — 2026-09-05

- Added a dedicated **Maps & spatial** tab inside 12A.1B GAMS-compatible ITA Studio.
- Added fully offline/API-key-free Greece rendering from bundled NUTS-2/NUTS-3 GeoJSON; no Mapbox/Google/OSM tiles are required.
- Added fine NUTS-3 boundary and coastline/island overlays while preserving NUTS-2 analytical fills.
- Added maps for allocation, budget utilisation, selected-project exposure, portfolio-score contribution, GREEN/GRAY/RED shares, dominant ITA class and Monte Carlo mean selection frequency when available.
- Added requested ITA colour semantics: GREEN #16A34A, GRAY #6B7280, RED #DC2626.
- Added publication exports: 600-dpi PNG, vector SVG, vector PDF and self-contained interactive HTML.
- Added explicit GAMS-region → NUTS-2 crosswalk. SYN2 `EP2` is preserved as the non-geographic EPANEK2 programme budget dimension and is never painted onto a Greek region.
- Added optional Moran's I/local spatial diagnostics for mapped continuous outputs.
- Added **GAMS diagnostics** tab with model/solver status, MIP node count, dual bound/gap where exposed by HiGHS, model size, active constraints, fixed variables, equation listing, X.l variable listing, source parameters and raw solver message.
- Added scenario-overlap/Jaccard and different-decision tables plus portfolio-score comparison plot.
- Extended reproducibility ZIP with solver diagnostics JSON, solver message, GAMS-style equation/variable listings and `.lst`-style text listing.
- No previous v5.7.1/v5.7.0/v5.6.1/v5.5.3 capability was removed or renamed.

# v5.7.1 — 2026-09-05

- Fixed GAMS regional auto-mapping so multiple regions are never silently assigned to the same numeric source column.
- Added explicit unmapped/duplicate-region validation and disables Solve until mappings are valid.
- Embedded the original supplied SYN2 540 GAMS input tables (`budget_syn2.prn`, `score_syn2.prn`, `sector_syn2.prn`) and GREEN/RED/GRAY sets as an exact-replication data source.
- Added a visible SYN2 source selector: original supplied GAMS inputs vs current application dataset.
- Increased caption/help-text contrast on dark surfaces.
- No existing capability removed.

# Changelog

## Version 5.7.0 — 5 September 2026

- Retained the complete v5.6.1 source tree and every v5.5.3 capability; this release is additive only.
- Added a separate `12A.1B GAMS-compatible ITA Studio` with visible GAMS-style sets, parameters, `X(p)`, `PORTFSCORE`, equations and model export.
- Added exact presets reconstructed from the supplied Evangelos Makryvelios GAMS sources: SYN2 540 Round 1–4 / No ITA and the 2,437-project R&D model with regional groups, sectors, interventions and round-specific Monte Carlo settings.
- Added SciPy/HiGHS execution for the GAMS-compatible model, preserving GAMS as an explicit source/reference/export route rather than requiring a commercial licence.
- Added hard GREEN/RED project fixing, GRAY/GREY effective-budget adjustments, region/sector/intervention ceilings, binding-constraint diagnostics and ITA-coloured Excel outputs.
- Added original supplied `.gms` files under `reference_gams/vangelis/` with in-app viewing/downloading.
- Added a visible session-only `LLM API Key` panel with Anthropic Claude and OpenAI-compatible providers; external LLM use is optional and invoked only after deterministic computation.
- Connected the user-key LLM co-pilot to the GAMS Studio and Research Command Chair for interpretation/drafting while keeping solver/statistical outputs independent of the LLM.
- Added final contrast guards so light controls use dark text and dark analytical surfaces use light text.


## Version 5.6.1 — September 2026

- Added a separate `12A.2 Expert respondent analytics` module without changing the retained v5.6.0 or v5.5.3 workflows.
- Added schema-agnostic mapping of respondent ID, ordered C1-Cn preference fields and an optional demographic/professional subgroup.
- Added complete-case or explicit median-imputation handling, invalid/negative-vector checks, duplicate-ID diagnostics and preservation of raw responses.
- Added respondent-level relative-weight distributions, non-parametric bootstrap mean intervals, Kendall's W concordance and Spearman dependence matrices.
- Added PCA structure, automatically selected K-means preference segments with silhouette diagnostics, respondent heatmaps and segment profiles.
- Added Kruskal-Wallis subgroup comparisons with Benjamini-Hochberg multiplicity control and epsilon-squared effect sizes.
- Added a one-click empirical weight bridge: Hybrid ITA-RW samples complete respondent vectors and progressively contracts them towards the empirical centre across rounds.
- Added complete XLSX/CSV/JSON respondent evidence exports. The exact confirmatory specifications remain to be reconciled with the two forthcoming papers rather than inferred in advance.

## Version 5.6.0 — September 2026

- Added a separate `12A.1 ITA / public-funding decision support` module while retaining every v5.5.3 module and workflow.
- Added ITA-PB rounds for pure-score allocation, C1 policy priority, an equity floor and full optimisation with call envelopes and beneficiary-category caps.
- Added Hybrid ITA-RW with simultaneous score and converging-weight uncertainty, reproducible Monte Carlo scenarios, Green/Gray/Red thresholds, frozen decisions and final gray-budget adjustment.
- Added exact SciPy/HiGHS binary portfolio optimisation and independent GAMS model/data export with project crosswalks and complete settings.
- Added modern project-by-round outcome matrices, Green/Gray/Red decision-flow infographics, score-versus-weight uncertainty bubbles and funding-envelope utilisation charts.
- Added regional allocation maps using the retained no-key Greek NUTS-2 boundaries, plus exploratory global/local Moran spatial diagnostics.
- Added per-project scorecards, regional and beneficiary profiles, observed/conventional/ITA comparisons and combined XLSX evidence export.
- Reserved the existing Hybrid ITA-RW sampling interface for later connection to validated respondent-level empirical distributions from the two expert papers; no unobserved distribution is inferred in this release.

## Version 5.5.3 — September 2026

- Added automatic workbook-sheet classification so README, dictionary, codebook, crosswalk, metadata, status-code and processing-log sheets are excluded from combination defaults while remaining manually selectable.
- Replaced the oversized unequal-row alert with a concise scientific warning and an expandable row-count table.
- Replaced the CARTO key-gated interactive basemap with OpenStreetMap, which requires no user API key.
- Retained v5.5.2 filename contrast, v5.5.1 column combination and all earlier analytical capabilities.

## Version 5.5.2 — September 2026

- Corrected uploaded filename and file-size contrast on Streamlit's light upload cards.
- Retained the v5.5.1 all-dataset selector and side-by-side column-combination workflow without analytical changes.
- Added an unmistakable v5.5.2 build marker so a stale v5.5.0 deployment can be identified immediately.

## Version 5.5.1 — September 2026

- Added `Combine columns side-by-side (by row order)` to the data intake console for datasets that describe the same ordered observations but hold different variables.
- Replaced the misleading single active-dataset selector in combination modes with a multi-dataset selector; all uploaded files and sheets are selected by default.
- Exposed the complete combined column set to every existing analytical module, including the Research Command Chair.
- Preserved duplicate column names using deterministic `__d2`, `__d3`, and subsequent source-order suffixes.
- Added an auditable one-based `__row_position__` field and explicit warnings for unequal source row counts.
- Retained separate-dataset, append-by-row, keyed-join and every v5.5.0 analytical/export capability.

## Version 5.5.0 — August 2026

- Added a one-screen Guided Chat Autopilot as the first Research Chair surface while retaining the complete original menu and advanced five-step workflow.
- Added selectable core, custom and all-question batches; every prepared question is editable in a white field with black type.
- Added explicit pre-execution verdicts: feasible/executed, statistically or logically invalid/incomplete, or not yet implemented in chat with a developer-contact route.
- Added direct Chair execution for normality, outlier, group-comparison, PCA, Cronbach reliability, advanced clustering, cross-validated prediction, ARIMA and panel-model requests when the required variable roles are mapped.
- Replaced instruction-only executive, data-quality, research-question, methodology and results responses with numerical findings and manuscript-ready prose.
- Added standalone interactive HTML charts and matching colour/black-and-white 600-dpi PNG, vector SVG and PDF figures to the Research Chair bundle.
- Added question-answer Word/Markdown files, figure index/commentary, plotted data and a self-contained interactive analytical report.
- Retained every v5.4.0 analytical module, export and specialist-menu route.

## Version 5.4.0 — August 2026

- Added an in-app, filterable library of 52 copy-ready prompts spanning all nineteen analytical modules.
- Added deterministic commands for Monte Carlo OLS with explicit method, seed and repetitions; permitted/prohibited conclusions; limitations; further-analysis priorities; research-question proposals; methods/results outlines; PDF-method replication audits; and equation/algorithm correctness checks.
- Added visible plain-language explanations to every table and chart, including purpose, reading instructions, pattern meaning, non-claims and next steps.
- Added `Output guide` worksheets to Excel exports and guide CSV/Markdown files to Research Chair, publication, OLS, Monte Carlo, clustering, predictive, panel and MCDA bundles.
- Added the complete prompt library to the Research Chair workbook and bundle.
- Retained every v5.3.2 analytical, UI, export and publication capability.

## Version 5.3.2 — August 2026

- Added a dedicated headline-finding pathway for questions such as “What is the most striking statistical result?”.
- Added a downloadable Ranked statistical findings table computed from sufficiently observed, varying, non-identifier numeric measures.
- Added direct reporting of effect magnitude, sample size and p-value; scientific meaning; data-quality warning; model-specification warning; publication-ready wording; and the next defensible analysis.
- Added explicit rejection of codes, identifiers and time fields as headline explanatory findings.
- Added in-app example questions and a visible answer-quality template.
- Forced typed, selected and placeholder text to high-contrast colours on light input fields.
- Retained every v5.2.1, v5.3.0 and v5.3.1 analytical and export capability.

## Version 5.3.1 — August 2026

- Fixed the empty Research Chair descriptive table: an unspecified model now describes every numeric variable in the saved analytical scope.
- Added a variable-level missingness table to every Research Chair execution.
- Changed free-form questions into deterministic data commands that compute descriptive, quality, correlation, longitudinal, OLS (when specified) and PDF-method evidence tables before answering.
- Changed “run the analysis as in paper” from generic advice to a computed paper-ready baseline with downloadable results.
- Added duplicate-question suppression and a clear-conversation control.
- Forced typed text and user messages to black on white/light-grey surfaces while retaining the existing dark theme.
- Retained every v5.2.1 and v5.3.0 module, estimator, export and publication bundle.

## Version 5.3.0 — August 2026

- Added Module 12B, the free/offline Research Command Chair.
- Added exact spreadsheet variable, row, filter and year-range scoping without changing the active source dataset.
- Added simultaneous PDF upload, local page-level extraction, document/page/keyword evidence selection and export.
- Added documented LaTeX equations and a restricted safe mathematical expression engine for derived variables.
- Added deterministic descriptive, longitudinal, correlation and HC3 OLS protocol execution.
- Added free built-in natural-language responses and optional local Ollama enhancement with no paid API requirement.
- Added Word/Markdown paper blueprints and a full reproducibility bundle containing filtered data, PDF evidence, protocol and results.
- Retained all eighteen v5.2.1 analytical modules and their outputs without removal.

## Version 5.2.1 — August 2026

- Added Module 12A, a reusable Dedicated MCDA Engine for projects, regions and other alternatives.
- Added auditable MAVT, TOPSIS and PROMETHEE-II scores and rankings.
- Added equal, user-defined, Entropy, CRITIC and AHP pairwise weighting with AHP consistency ratio.
- Added explicit maximise/minimise criterion directions and missing-data controls.
- Added one-at-a-time weight sensitivity, method rank correlations and reproducible Monte Carlo rank acceptability.
- Added complete MCDA XLSX/CSV tables and colour/black-and-white publication bundles in PNG 600 dpi, SVG and PDF.
- Rebuilt the complete application documentation: consolidated DOCX/PDF technical report, searchable Markdown manual, quick start, MCDA guide, deployment/operations guide, validation protocol and requirements-coverage matrix.
- Added in-app access to the documentation library through Module 13.
- Retained every v5.2 capability without removal or behavioural regression.

## Version 5.2 — August 2026

- Changed non-purple dark narrative text on dark dashboard surfaces to white.
- Changed blue text on green/teal notification surfaces to white.
- Retained the light-purple widget labels, all backgrounds, analytical functions and existing capabilities unchanged.

## Version 4.1 — August 2026

- Applied a global high-contrast light-purple style to Streamlit widget labels across every dark dashboard panel.
- Preserved dark input text on light controls and the existing cyan upload treatment for maximum readability.
- Added a smoke-test guard for the label contrast rule so future UI changes do not silently reintroduce the issue.

## Version 4.0 — August 2026

- Rebuilt the interface as a dark postdoctoral analytical command centre and corrected the low-visibility upload control.
- Added one-dimensional absorption clustering and multivariate K-means, hierarchical, Gaussian-mixture and DBSCAN workflows.
- Added automatic k selection, silhouette, Calinski–Harabasz, Davies–Bouldin, profiles, projections and clustering publication bundles.
- Added ten-refit perturbation stability diagnostics using the Adjusted Rand Index.
- Added pooled OLS, two-way fixed effects, random effects and Hausman panel-model comparison.
- Added seven-model cross-validated predictive comparison and permutation importance.
- Added Huber robust regression and Gamma log-link regression.
- Added effect sizes to group tests and direct downloadable effect-size visualisations.
- Added robust IQR/MAD outlier surveillance and a consolidated data-audit workbook without automatic deletion or winsorisation.
- Expanded figure HTML/data exports, explanations and publication bundles across the app while retaining every v3 capability.

## Version 3.0 — August 2026

- Made OLS explicit in navigation and rebuilt the econometric screen as an OLS Studio with estimator guidance.
- Added wild-bootstrap, residual-bootstrap and parametric-normal Monte Carlo OLS with reproducible seeds and full draw exports.
- Added stochastic R&D portfolio selection under uncertain cost and benefit, including selection probabilities and downside distributions.
- Added downloadable OLS and Monte Carlo publication bundles in colour and black-and-white, 600-dpi PNG, SVG and PDF.
- Added coefficient-forest and residual plots, interactive HTML downloads and plotted-data downloads.
- Added an analysis navigator, module-by-module operating instructions and contextual interpretation comments.
- Incorporated methodological evidence from the additional Makryvelios manuscripts and supplied research outputs without removing any v2 capability.

## Version 2.0 — August 2026

- Rebuilt the original single-dataset Streamlit dashboard as a multi-file research workbench.
- Added concurrent XLSX/XLS/CSV/TSV and all-sheet ingestion, append and keyed joins.
- Preserved the supplied original application in `archive/original_app.py`.
- Bundled and auto-detected the 3,259 × 83 R&D reference workbook.
- Recovered EE1–EE9, nineteen hypotheses, project-level Attica/country specifications and region–year panel models from the supplied sources.
- Added an explicit Antonis Tritsis and cross-programme research catalogue.
- Added detailed statistics, hypothesis tests, robust econometrics, IV/2SLS, DiD, regularisation, high-dimensional vectorised OLS, PCA, clustering, reliability and time-series modules.
- Added offline official GISCO NUTS-2/NUTS-3 Greece boundaries, Moran/LISA diagnostics and bilingual region matching.
- Added 600-dpi and vector colour/black-and-white figure and map exports.
- Added exact table workbooks, CSV outputs and self-contained HTML/JavaScript reports.
- Added optional R replication and constrained allocation/scenario tools.
- Added deterministic unit tests and a thirteen-module Streamlit render smoke test.

## v5.8.6 — Groq GPT-OSS reasoning/output reliability hotfix (2026-09-06)
- Preserves all v5.8.5 functionality; additive/compatibility-only change.
- Fixes Groq GPT-OSS HTTP-200 responses where reasoning consumed the completion budget and `message.content` was empty.
- Uses `max_completion_tokens` for Groq GPT-OSS, explicit `reasoning_effort`, and `include_reasoning=false` for structured research tasks.
- Adds a visible Groq GPT-OSS reasoning-effort selector (Low/Medium/High), default Low; no provider or model fallback occurs.
- Normalises nested JSON schemas for Groq strict Structured Outputs (`additionalProperties=false`, all object properties required).
- Keeps GPT-OSS instructions in the user turn, following Groq reasoning-model guidance.
- Empty-content errors now report finish reason, completion usage and whether reasoning was present instead of a generic 'no text content' message.
