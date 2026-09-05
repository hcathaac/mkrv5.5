# Frontier Methods Laboratory and Agentic Research Mode — v5.8.0

## Scope

Version 5.8.0 is strictly additive over v5.7.2. No earlier analytical, GAMS/ITA, respondent, GIS, Research Command Chair, LLM or export capability is removed.

## 12C — Frontier Methods Laboratory

### Pareto / multi-objective / robust optimisation

Maps a project identifier, two to four objectives to maximise, project cost and a budget. The engine enumerates a deterministic simplex of objective weights, solves the binary portfolio problem with SciPy/HiGHS and retains only non-dominated portfolios. Optional objective-uncertainty columns apply lower-confidence robust penalties and an optional cost-uncertainty field inflates resource use. Outputs include the Pareto frontier, project selection frequency and downloadable Excel evidence.

### Causal inference

Implements a cross-fitted augmented inverse-probability weighted (AIPW) average treatment effect for a binary treatment. Propensity and outcome nuisance models are fit out-of-fold. The module reports the ATE, influence-function standard error and interval, propensity overlap, unit diagnostics and standardised mean differences before/after IPW. The interface explicitly states the identification assumptions and never infers causal validity from the dataset alone.

### Bayesian modelling

Provides a fully offline Gaussian Bayesian linear-regression laboratory using the exact conjugate Normal–Inverse-Gamma posterior. It generates posterior coefficient draws, posterior probabilities of coefficient direction, 95% intervals and posterior-predictive distributions with coverage/RMSE/MAE diagnostics. No external AI or probabilistic-programming service is required.

### SHAP / explainable ML

Fits a random-forest regression/classification model and, when SHAP is installed, uses TreeExplainer for global mean absolute SHAP importance and local feature contributions. A deterministic permutation/local-perturbation fallback prevents loss of functionality if SHAP cannot load. Numerical prediction remains in scikit-learn.

### DuckDB / Arrow / Parquet

Adds a read-only DuckDB query surface over the active dataset, Arrow IPC export and compressed Parquet export. v5.8.0 also expands the intake layer to accept Parquet, Feather and Arrow IPC files in addition to the retained Excel/CSV/TSV formats. These layers are additive accelerators/interchange formats; the existing pandas workflows remain unchanged.

## 12D — Agentic Research Mode

Agentic Research Mode is a standalone, approval-gated end-to-end research runner. It is deliberately offline-first.

### Without an LLM API

The mode can:

1. index uploaded literature PDFs locally by document and page;
2. detect DOI/URL/year strings without inventing missing bibliography metadata;
3. extract concise page-level evidence notes and frequent literature terms;
4. generate up to 150 candidate research questions in a single batch from the active variables and literature themes;
5. build a transparent ordered research plan;
6. require explicit user approval before running the plan;
7. audit the data and run descriptive statistics, correlations, HC3 OLS where mapped, group tests, PCA and clustering where feasible;
8. create offline evidence-grounded discussion, conclusions and limitations;
9. assemble a near-submission package containing Word, Excel, JSON, HTML, 600-dpi/vector figures and a reproducibility manifest.

### With a user-supplied LLM API key

The same deterministic run remains the evidential source. The LLM receives a compact computed-evidence summary only after the user explicitly requests synthesis. It can refine discussion, conclusions, next-analysis recommendations and manuscript prose, but it does not replace the numerical engines, change results, select GAMS portfolios or bypass approval gates.

## Submission-package boundary

The generated bundle is intentionally close to submission-ready to remove repetitive analytical labour, but it is not automatically submission-safe. A human researcher must verify bibliography metadata, quotations, causal wording, final model choices, journal/university formatting, disclosure requirements and all substantive conclusions.

## Preservation

v5.8.0 retains every v5.7.2 module and adds two new top-level modules only: **12C Frontier methods laboratory** and **12D Agentic Research Mode**.
