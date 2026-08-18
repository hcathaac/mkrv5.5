# Makryvelios Research Analytics & Econometrics Command Centre

## Complete Technical and User Documentation - Version 5.2.1

**Software type:** Streamlit analytical dashboard  
**Primary runtime:** Python 3.11/3.12  
**Auxiliary replication layer:** R, where independently installed  
**Interface technologies:** Streamlit, Plotly and self-contained HTML/JavaScript reports  
**Release:** 16 August 2026  
**Entry point:** `makryvelios_dashboard_v2/app.py`

---

## Table of contents

1. Purpose and scope
2. System architecture
3. Installation and deployment
4. Data ingestion and preparation
5. Module-by-module user manual
6. Econometric specification and interpretation
7. Dedicated MCDA methodology
8. Publication outputs
9. Validation and quality assurance
10. Performance and scalability
11. Error handling and troubleshooting
12. Reproducibility protocol
13. Requirements coverage matrix
14. Limitations and research-governance rules
15. Release identification

---

## 1. Purpose and scope

The Makryvelios Research Analytics & Econometrics Command Centre is a reusable research workbench for project-level, regional, panel, spatial and policy datasets. It was initially structured around the supplied Greek R&D-project evidence and the planned «Αντώνης Τρίτσης» dataset. Version 5.2.1 extends the same architecture to renewable-energy portfolios and other decision problems through a dedicated multi-criteria decision-analysis engine.

The application is designed to support the complete analytical cycle: data ingestion, structural audit, descriptive analysis, hypothesis testing, econometric estimation, uncertainty analysis, high-dimensional screening, panel analysis, geographical analysis, time-series and multivariate analysis, clustering, predictive validation, scenario modelling, constrained allocation, multi-criteria ranking, interpretation and publication export.

The software does not automate substantive judgement. It exposes assumptions, samples, transformations, diagnostics, ranking rules and robustness evidence so that the researcher can defend each analytical decision. Outputs are suitable for exploratory work, technical reporting and manuscript preparation, subject to the data-quality and identification restrictions documented below.

### 1.1 Coverage statement

Version 5.2.1 retains every capability available in version 5.2. No estimator, upload mode, diagnostic, map, report, graph or download mechanism has been removed. The release adds a dedicated MCDA module, renewable-energy documentation, additional validation tests and expanded technical documentation.

### 1.2 Intended users

- Researchers analysing programme, project, municipal, regional or national datasets.
- Postgraduate and postdoctoral researchers requiring auditable analytical workflows.
- Policy analysts comparing regions, programmes, technologies or investment alternatives.
- Research teams preparing tables and figures for journal manuscripts.
- Analysts requiring a Python-first application with optional R replication.

### 1.3 Appropriate analytical units

The application can operate on project, organisation, municipality, technology, programme, region, region-year or country-year records. The unit of analysis must be identified before estimation. Mixing project-level outcomes with region-level explanatory variables without accounting for clustering or repeated regional values can understate uncertainty and generate misleading significance tests.

---

## 2. System architecture

The application uses a modular architecture in which calculation code is separated from the Streamlit interface. This separation enables deterministic testing, independent reuse and transparent export.

| Component | Responsibility |
|---|---|
| `app.py` | Streamlit user interface, navigation, state management and module orchestration |
| `analytics_core.py` | File ingestion, cleaning, descriptive statistics, tests, econometrics, simulation, time series and exports |
| `advanced_analytics.py` | Validated clustering, predictive comparisons and panel-model suite |
| `mcda.py` | MAVT, TOPSIS, PROMETHEE II, weighting, sensitivity, rank acceptability and MCDA publication outputs |
| `legacy_rd.py` | Recognition of the original R&D schema, region-year construction and EE1-EE9 presets |
| `mapping.py` | Greek NUTS matching, GIS aggregation, choropleths and Moran/LISA diagnostics |
| `visuals.py` | Interactive figures and colour/black-and-white publication bundles |
| `reporting.py` | Portable self-contained HTML/JavaScript analytical report |
| `r_engine.R` | Optional independent R replication for supported configurations |
| `data/` | Reference workbook, Greece lookup and offline NUTS-2/NUTS-3 boundaries |
| `research_questions.csv` | R&D and Antonis Tritsis research-question catalogue |
| `research_hypotheses.csv` | Recovered R&D hypothesis catalogue |
| `source_evidence_catalogue.csv` | Source-to-method provenance and implementation traceability |
| `tests/` | Deterministic calculation, rendering and interaction tests |

### 2.1 Execution model

The browser communicates with the Streamlit Python process. Uploaded files are held within the active Streamlit session and are not sent to an external analytical API. Calculations execute on the host running the application. The HTML export embeds result tables and JavaScript required to display the report; it is a portable result snapshot, not a live copy of the analytical engine.

### 2.2 Python and R responsibilities

Python is the production engine and provides the full application. R is auxiliary and optional. Absence of `Rscript` does not disable the dashboard. R should be used for independent replication when a second implementation is methodologically useful; it is not a hidden dependency and it does not replace the tested Python calculations.

### 2.3 Session state

Model outputs are stored in Streamlit session state so users can move between output panels and produce consolidated downloads without rerunning every calculation. Uploading new files or changing the active dataset requires the user to verify that retained results still refer to the intended data. A browser refresh can clear session-held outputs.

---

## 3. Installation and deployment

### 3.1 Local installation

Python 3.11 or 3.12 is recommended.

```bash
cd makryvelios_dashboard_v2
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and run the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default local address is `http://localhost:8501`.

### 3.2 GitHub and Streamlit Community Cloud

1. Extract the release ZIP.
2. Upload the complete `makryvelios_dashboard_v2` folder to the GitHub repository.
3. Confirm that `app.py`, `mcda.py`, all other Python modules, `requirements.txt`, `.streamlit/config.toml`, catalogue CSVs and required `data/` files are present.
4. Commit to the branch used by Streamlit, normally `main`.
5. In Streamlit Community Cloud select the repository and branch.
6. Use `makryvelios_dashboard_v2/app.py` as the main file when the folder sits below the repository root. Use `app.py` only when the folder contents themselves are at the root.
7. Deploy or reboot the existing application.
8. Inspect the build log for missing packages, incorrect paths or data-file case mismatches.

GitHub stores the source code; it does not execute Streamlit. A missing “Deploy” button on the GitHub repository page is normal. Deployment is initiated from Streamlit Community Cloud.

### 3.3 Confidentiality

The release can contain a reference R&D workbook. It must not remain in a public repository if the records are confidential or restricted. Remove the workbook from `data/`, use a private repository, or upload it through the application at runtime. Uploaded datasets are not deliberately transmitted to an external analytical service by the application.

### 3.4 Updating an existing deployment

The compatibility folder remains `makryvelios_dashboard_v2`; consequently, an existing main-file path does not need to change. Replace matching files, ensure the new `mcda.py` and `documentation/` folder are included, commit, then reboot the Streamlit app if automatic rebuilding does not begin.

---

## 4. Data ingestion and preparation

### 4.1 Supported files

The sidebar accepts multiple `.xlsx`, `.xlsm`, `.xls`, `.csv` and `.tsv` files simultaneously. Excel workbooks may be read from the first sheet only or from every sheet. Each file-sheet combination receives a source label.

CSV decoding is attempted with UTF-8 BOM, UTF-8, Greek Windows CP1253 and Latin-1 encodings. Delimiters are inferred. Spreadsheet support depends on the engines declared in `requirements.txt`.

### 4.2 Column-name normalisation

Normalisation trims spaces, replaces unsupported punctuation with underscores and generates unique names for duplicates. It does not modify cell values. Normalisation is recommended for modelling because formula construction and cross-file matching are more reliable with syntactically safe, unique names.

### 4.3 Embedded-header correction

The original R&D workbook contains a numeric index row above the substantive variable names. The ingestion engine detects the pattern and promotes the first data row when the current headings are predominantly numeric, the candidate row is predominantly textual and its non-missing values are unique. The rule is deliberately constrained so ordinary first observations are not promoted mechanically.

### 4.4 Multiple-dataset relationships

**Keep datasets separate** analyses one selected file or sheet at a time. Use this mode when sources have different units, definitions or periods.

**Append rows** performs a union by column name. Use it only when rows describe the same unit and variables have compatible definitions. Missing columns are retained and filled with missing values.

**Join datasets on keys** merges sources through one or more common keys. The interface supports outer, left, inner and right joins. Before joining, confirm key uniqueness, data type, case, spelling, time reference and analytical grain. A many-to-many join can multiply records and invalidate every subsequent calculation.

### 4.5 Data audit

The Data hub reports rows, columns, data types, inferred analytical roles, missingness, unique values, zero variance, duplicates and robust outlier indicators. The audit identifies conditions that require review; it does not silently delete, winsorise or impute the source data.

### 4.6 Minimum pre-analysis checks

1. Identify the unit of analysis.
2. Confirm that each row represents one unit-period combination where required.
3. Check duplicated identifiers and join keys.
4. Verify that percentages use a consistent scale, either 0-1 or 0-100.
5. Distinguish genuine zero from missing or inapplicable values.
6. Verify dates, project start/end years and region labels.
7. Inspect missingness by outcome, exposure, region and period.
8. Confirm that repeated regional indicators are not treated as independent project measurements without appropriate clustered or multilevel inference.

---

## 5. Module-by-module user manual

### 5.1 Module 1 - Executive overview

This page summarises the active dataset, including record count, variable count, numeric/categorical composition and research-question coverage. The method navigator maps analytical objectives to the appropriate module and principal outputs. Missingness and variable-composition charts are downloadable. Detection of the original R&D schema activates dedicated messages and regional presets.

Use this page to select the analytical route. It does not estimate causal effects.

### 5.2 Module 2 - Data hub & audit

This module displays file/sheet inventory, quality checks, data dictionary, record preview, missingness and outlier surveillance. Cleaned analytical views and audit tables can be downloaded. A clean interface does not prove that concepts are validly measured; substantive coding decisions must still be documented.

### 5.3 Module 3 - Research questions

The research catalogue contains the recovered R&D questions and hypotheses, reconstructed Antonis Tritsis questions and cross-programme extensions. Filters help the researcher identify outcome, explanatory variables, unit of analysis and suitable methods. Reconstructed questions are labelled as such and are not represented as verbatim source content.

### 5.4 Module 4 - Descriptive statistics

Numeric summaries include count, missingness, mean, standard error, standard deviation, minimum, quartiles, maximum, skewness, kurtosis and coefficient of variation. Categorical summaries report frequencies and proportions. Pearson, Spearman and Kendall correlations are available with p-values. Correlation is symmetric association and does not establish direction, mechanism or causality.

### 5.5 Module 5 - Hypothesis tests

The module provides Welch t-tests, Mann-Whitney tests, one-way ANOVA, Kruskal-Wallis tests, Levene tests, chi-square tests with Cramér’s V, Shapiro-Wilk, D’Agostino K² and Anderson-Darling diagnostics. Effect sizes include Hedges’ g, rank-biserial correlation, eta-squared, epsilon-squared and Cramér’s V where applicable.

Normality tests become highly sensitive in large samples. Distribution plots, robust methods and the consequence for the estimand should be considered alongside p-values. Multiple testing requires correction or an explicitly pre-specified primary hypothesis.

### 5.6 Module 6 - OLS & econometric laboratory

OLS is the default transparent estimator for continuous outcomes. The laboratory also supports WLS, Huber robust regression, logit, probit, Poisson, negative-binomial GLM, fractional logit, Gamma log-link and quantile regression. Categorical controls and fixed effects can be encoded. Covariance options include conventional, HC0-HC3, HAC and clustered standard errors.

Diagnostics include VIF, Breusch-Pagan, White, RESET, Durbin-Watson, Jarque-Bera, omnibus normality, Cook’s distance and estimator-specific fit/dispersion information. The output includes exact coefficient tables, fit statistics, diagnostic tables, fitted/residual evidence and publication bundles.

Interpret coefficients on their estimator-specific scale. For binary and count models, exponentiated coefficients are useful only when their interpretation is appropriate. Robust standard errors adjust inference, not omitted-variable bias, reverse causality, measurement error or sample selection.

### 5.7 Module 6A - Monte Carlo & uncertainty

The OLS simulation laboratory provides wild bootstrap, residual bootstrap and parametric-normal simulation. It exports every coefficient draw, simulation mean, empirical bias, Monte Carlo standard error, sign probability and percentile interval. A fixed seed ensures reproducibility.

The R&D portfolio simulation propagates uncertainty in costs and benefits under a budget envelope. It reports selection probabilities, portfolio cost/benefit distributions and downside evidence. These probabilities describe the specified stochastic model, not the objective probability that a project succeeds.

### 5.8 Module 7 - 1,000 × 1,000 batch engine

The interface accepts up to 1,000 dependent and 1,000 independent variables. A vectorised SVD/pseudoinverse engine performs large OLS screens and applies Benjamini-Hochberg or Bonferroni correction. The engine is a discovery and screening facility, not an exemption from sample-size, rank or identification requirements.

When predictors approach or exceed the number of independent observations, coefficients can be non-unique or unstable. Shortlisted relationships must be re-estimated in the detailed econometric laboratory with theory-led controls and defensible covariance choices.

### 5.9 Module 8 - Original R&D regional panel

The R&D compatibility layer constructs region-year data using start-year, end-year or active-year project allocation. It provides recovered EE1-EE9 presets, regional summaries, project counts, expenditure/absorption aggregates and innovation indicators. Results depend materially on the year-allocation convention and must report it.

### 5.10 Module 8A - Panel model laboratory

The panel suite compares pooled OLS, two-way fixed effects and random effects. It supports entity-clustered covariance and a Hausman FE-RE specification diagnostic. Duplicate entity-time rows are aggregated through the selected rule before estimation.

Fixed effects identify relationships from within-entity variation and absorb time-invariant entity characteristics. Random effects require the entity effect to be uncorrelated with regressors. The Hausman result is one diagnostic, not a substitute for substantive model design.

### 5.11 Module 9 - Detailed Greece GIS

The GIS module uses official Eurostat GISCO NUTS 2024 boundaries at NUTS-2 and NUTS-3 levels. A bilingual lookup matches common Greek and English region labels. Users select a geographical identifier, aggregation rule and metric. Interactive choropleths, 600-dpi static maps and vector outputs are available in colour and monochrome.

Global Moran’s I assesses overall spatial autocorrelation through K-nearest-neighbour weights and permutation inference. Local output provides a LISA-style cluster diagnostic. Results depend on the spatial weight matrix, geographical level, aggregation rule, missing regions and the modifiable areal-unit problem.

### 5.12 Module 10 - Time series & multivariate

The module includes ADF and KPSS stationarity tests, Granger predictability, ARIMA forecasting, PCA, legacy standardised k-means and Cronbach’s alpha/item diagnostics. ADF and KPSS use different null hypotheses and should be interpreted jointly. Granger predictability is temporal predictive content, not structural causality. ARIMA forecasts depend on order, time regularity and stability of the generating process.

PCA loadings describe linear components of standardised variables. Component signs are arbitrary. Cronbach’s alpha assesses internal consistency under assumptions that do not prove unidimensionality or construct validity.

### 5.13 Module 10A - Advanced clustering & segmentation

The advanced clustering module supports one-variable absorption clustering and multivariate K-means, hierarchical agglomerative clustering, Gaussian mixture models and DBSCAN. Scaling choices include standard, robust and min-max transformations. Automatic cluster-number selection uses silhouette evidence with supporting Calinski-Harabasz and Davies-Bouldin measures. Ten-refit perturbation stability is summarised by the adjusted Rand index.

Outputs include assignments, substantive cluster profiles, internal validation, PCA or one-dimensional projections and publication bundles. Cluster labels are descriptive and arbitrary. Stability, interpretability, sample size and sensitivity to scaling must be reported; clustering does not establish latent causal classes.

### 5.14 Module 10B - Predictive model laboratory

Seven continuous-outcome models are compared: OLS, Ridge, Lasso, Elastic Net, random forest, extra trees and gradient boosting. Cross-validated out-of-fold RMSE, MAE and R² provide model-comparison evidence. Permutation importance summarises the predictive contribution of variables under the fitted model and observed correlation structure.

Data leakage must be prevented. Variables measured after the outcome, duplicates of the outcome, future information and identifiers must not enter predictors. Predictive importance is not a causal effect.

### 5.15 Module 11 - Publication figures & HTML report

Users can create interactive Plotly figures, download plotted data and export publication packages. Static packages contain colour and black-and-white versions in 600-dpi PNG, vector SVG and vector PDF formats. The consolidated HTML report contains dataset information, selected results, explanatory comments and embedded JavaScript required for portable viewing.

The researcher remains responsible for journal dimensions, font requirements, caption style, decimal precision and accessibility. Vector output is preferred when the journal accepts it.

### 5.16 Module 12 - Scenario & allocation engine

The econometric shock tool applies an OLS partial slope to a stated percentage change in a driver and summarises projected outcome differences. It is a ceteris-paribus association-based scenario, not a causal forecast.

The allocation tool uses linear programming to distribute an additional resource envelope by a stated need/benefit score subject to a maximum share. Policy use requires explicit eligibility, floors, costs, capacity, fairness and legal constraints. Omitted constraints are not inferred automatically.

### 5.17 Module 12A - Dedicated MCDA engine

The MCDA module ranks projects, regions, technologies or other alternatives using two to fifty numeric criteria. Up to fifteen criteria can use AHP pairwise comparisons. Larger criterion sets should use equal, user-defined, Entropy or CRITIC weights because AHP judgement burden grows quadratically.

Users select an alternative identifier, criteria, maximise/minimise direction, missing-data rule, weight method, ranking methods, primary robustness method, sensitivity range, Monte Carlo weight draws, concentration and seed. Outputs include rankings, method scores, normalised matrix, weights, AHP diagnostics, rank correlations, sensitivity, rank acceptability and publication files.

Full methodological details are provided in Section 7.

### 5.18 Module 13 - Methods & reproducibility

The final module catalogues analytical families, implementation choices and source evidence. It reports whether `Rscript` is installed and provides the commands needed to reproduce the Python environment. The source-to-method catalogue distinguishes recovered methods from new postdoctoral extensions.

---

## 6. Econometric specification and interpretation

### 6.1 OLS

For outcome vector y and design matrix X, OLS estimates:

**β̂ = (XᵀX)⁻¹Xᵀy**

or a numerically appropriate pseudoinverse when the matrix is not of full rank. A coefficient is the expected conditional difference in the outcome associated with a one-unit change in a predictor, holding included variables constant. Causal language requires an identification strategy that supports exogeneity.

### 6.2 Robust covariance

HC estimators adjust standard errors for heteroskedasticity. HAC addresses specified serial dependence. Clustered covariance permits within-cluster error dependence when a sufficient number of defensible clusters exists. None of these procedures repairs a biased coefficient caused by omitted confounding or endogenous regressors.

### 6.3 Binary, fractional and count outcomes

Logit and probit are designed for binary outcomes. Fractional logit is suitable for outcomes bounded in [0,1], including proportions when boundary values are present. Poisson and negative-binomial models are designed for non-negative counts. Gamma log-link requires strictly positive continuous outcomes. Distributional assumptions, link functions, exposure/offset decisions and dispersion must match the research design.

### 6.4 IV/2SLS

The IV implementation estimates an endogenous predictor using excluded instruments and then estimates the outcome equation with the first-stage fitted component. Instrument relevance is not enough: the exclusion restriction and independence assumptions require substantive defence. Weak instruments can make inference unreliable.

### 6.5 Difference-in-differences

The DiD coefficient is the treatment-by-post interaction. Its causal interpretation requires parallel counterfactual trends, stable composition, absence of anticipatory effects and no contemporaneous differential shocks. With multiple periods or staggered adoption, the simple two-way interaction may be inadequate.

### 6.6 Regularisation

Ridge, Lasso and Elastic Net can stabilise prediction and select predictors, but their shrunken coefficients do not have the same inferential interpretation as pre-specified OLS coefficients. Hyperparameters and preprocessing must be selected without using test outcomes.

### 6.7 Multiple testing

Benjamini-Hochberg controls the expected false-discovery rate under its conditions; Bonferroni controls family-wise error more conservatively. Adjustment reduces false positives but does not solve specification search, data leakage or weak theoretical justification.

---

## 7. Dedicated MCDA methodology

### 7.1 Decision matrix

Let xᵢⱼ denote performance of alternative i on criterion j. Each criterion is oriented so that larger normalised values are preferred. Benefit criteria use min-max normalisation; cost criteria reverse the scale. A constant criterion receives a neutral value of 0.5 and is flagged because it cannot discriminate between alternatives.

The default missing-data rule uses criterion medians. Complete-case analysis is also available. Median imputation preserves alternatives but understates uncertainty and can compress differentiation. The rule must be reported.

### 7.2 Weights

Weights are finite, non-negative and normalised to sum to one.

**Equal weights** treat all criteria symmetrically.

**User-defined weights** encode an explicit policy or expert preference structure.

**Entropy weights** assign greater weight to criteria with more observed diversification after normalisation.

**CRITIC weights** combine criterion dispersion with conflict/redundancy measured through correlations.

**AHP weights** derive the principal eigenvector of a positive reciprocal pairwise comparison matrix. The consistency index is:

**CI = (λmax - m)/(m - 1)**

and the consistency ratio is CR = CI/RI, where RI is Saaty’s random index for m criteria. A value above approximately 0.10 normally requires reconsideration, though the threshold is a diagnostic convention rather than a proof of validity.

Objective weights describe structure in the observed data; they do not replace ethical, strategic or policy judgement.

### 7.3 MAVT

The implemented additive value model calculates:

**Vᵢ = Σⱼ wⱼvⱼ(xᵢⱼ)**

where vⱼ is the oriented min-max value. MAVT is fully compensatory: sufficiently strong performance on one criterion can offset poor performance on another. This property must be appropriate for the decision.

### 7.4 TOPSIS

TOPSIS ranks alternatives by relative closeness to the weighted ideal and distance from the anti-ideal:

**Cᵢ = Dᵢ⁻/(Dᵢ⁺ + Dᵢ⁻)**

A larger score indicates greater closeness to the ideal. Results can be sensitive to normalisation, criterion correlation and the set of alternatives.

### 7.5 PROMETHEE II

PROMETHEE II calculates weighted pairwise preference differences, positive leaving flow, negative entering flow and net flow. The net flow creates a complete ranking. The current transparent implementation uses a linear preference on the normalised difference without separate indifference or preference thresholds. Threshold-based elicitation can be introduced in a future domain-specific version when validated expert judgements are supplied.

### 7.6 Consensus and method agreement

When multiple methods are selected, percentile-based method scores produce a consensus indicator. Spearman rank correlations reveal the extent of agreement. Large disagreements should be reported and explained; they indicate sensitivity to compensability, distance or pairwise-preference structure and should not be concealed by an average rank.

### 7.7 Weight sensitivity

Each criterion weight is reduced and increased by the selected percentage while remaining weights are proportionally renormalised. The engine reports the new top alternative and Spearman stability. This is a one-at-a-time local sensitivity analysis and does not exhaust all possible joint preference structures.

### 7.8 Monte Carlo rank acceptability

The engine draws criterion-weight vectors from a Dirichlet distribution centred on the baseline weights. The concentration parameter controls dispersion: large values keep draws close to baseline weights; small values permit wider variation. For MAVT or TOPSIS, the application reports probability of rank 1, probability of top 3, expected rank, rank standard deviation and rank-by-rank acceptability.

These probabilities quantify ranking robustness to the specified weight distribution. They are not probabilities of financial, technical, environmental or implementation success.

### 7.9 Renewable-energy use

The future 2,000-project renewable-energy dataset can be processed without a new application. Candidate criteria can include capacity factor, internal rate of return, payback, return on investment, benefit-cost ratio, employment, regional-development contribution, avoided emissions, land requirements, grid constraints, licensing delays and project risk. Direction, unit, time basis and treatment of technology-specific structural differences must be documented.

Spatial clustering, econometric modelling and MCDA answer different questions. Clustering identifies empirical groupings; econometrics estimates conditional associations or causal effects under assumptions; MCDA ranks alternatives under an explicit preference model. They should be used as complementary layers rather than interchangeable procedures.

---

## 8. Publication outputs

### 8.1 Tables

Displayed tables can be downloaded as CSV or grouped into Excel workbooks. Exports preserve the exact numerical output used by the interface. Researchers should retain full precision in archival tables and apply journal-specific rounding only in manuscript-facing tables.

### 8.2 Figures

Interactive Plotly charts support exploration and browser-level image capture. Explicit publication bundles provide the archival outputs: colour and black-and-white 600-dpi PNG plus SVG and PDF vector forms. Black-and-white versions use monochrome palettes and, where implemented, hatching or contrast appropriate for print.

### 8.3 MCDA publication bundle

The MCDA ZIP contains ranking, weights, criterion-performance and rank-acceptability figures in colour and black-and-white formats. It also contains ranking, weight, normalised matrix, method-correlation, sensitivity, acceptability and diagnostic CSV files plus a README describing interpretation.

### 8.4 HTML report

The self-contained report consolidates dataset metadata, selected analytical tables and explanatory comments. It is suitable for circulation and archiving but does not contain the computational engine. A changed input dataset requires a new report.

### 8.5 Reporting checklist

- Name the dataset version and extraction date.
- State unit of analysis, period and exclusions.
- Report transformations and missing-data rules.
- Identify outcome, predictors, fixed effects and covariance estimator.
- Report sample size used by each model.
- Present effect sizes and confidence intervals, not only p-values.
- Document multiplicity correction where relevant.
- Report model diagnostics and robustness checks.
- For spatial models, state geography and weight matrix.
- For clustering, state scaling, method, selected cluster count and stability.
- For MCDA, state criteria, directions, weights, method and sensitivity design.
- Separate association, prediction, scenario and causal claims.

---

## 9. Validation and quality assurance

### 9.1 Calculation tests

The deterministic test suite covers header promotion, descriptive statistics, correlations, outlier summaries, OLS recovery, batch OLS, Excel export, Greek region aliases, Moran diagnostics, R&D panel construction, IV/2SLS, DiD, regularisation, reliability, Monte Carlo reproducibility, portfolio probabilities, publication bundles, advanced clustering, predictive comparison, panel models, Huber/Gamma estimators, MCDA reproducibility and AHP consistency.

### 9.2 Interface tests

The Streamlit smoke test renders all modules against the bundled workbook. Interaction tests exercise OLS, Monte Carlo, advanced clustering, predictive comparison, panel models and the dedicated MCDA workflow. These tests detect exceptions and confirm the appearance of principal output panels.

### 9.3 Release checks

Before distribution:

1. Compile all Python modules.
2. Run calculation tests in an environment built from `requirements.txt`.
3. Run Streamlit smoke and interaction tests.
4. Test multi-file upload, append and keyed join with representative files.
5. Verify colour contrast and upload-control visibility.
6. Verify every download button returns a non-empty file.
7. Open at least one Excel, HTML and publication ZIP output.
8. Verify map rendering with offline and fetched boundaries.
9. Confirm `VERSION.txt`, header, changelog and documentation identify the same release.
10. Test the packaged ZIP after extraction, not only the source directory.

### 9.4 Known validation boundary

Software tests establish consistency of implementation; they do not validate every future dataset, research design or substantive conclusion. Numerical equality with previous Stata/R results requires identical input data, filters, coding, allocations, fixed effects, estimator options and covariance assumptions.

---

## 10. Performance and scalability

The batch selector supports 1,000 outcomes and 1,000 predictors, but computational availability is not statistical identification. Memory and runtime depend on rows, columns, missingness and chosen method.

PROMETHEE II uses pairwise alternative comparisons and therefore grows approximately with the square of the number of alternatives. Around 2,000 alternatives is feasible on an adequately provisioned host, but substantially larger samples or many criteria may require batching or a more memory-efficient preference implementation.

Monte Carlo runtime grows with the number of draws, alternatives and methods. Begin with 100-500 draws for configuration checks, then run 1,000-5,000 for the retained specification. Publication bundles render high-resolution figures and may take longer than interactive charts.

Streamlit Community Cloud resource limits can be lower than a research workstation. For large confidential datasets or repeated simulations, local deployment, a private server or a higher-resource hosting plan is preferable.

---

## 11. Error handling and troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| App does not deploy | Wrong main-file path | Use `makryvelios_dashboard_v2/app.py` when the folder is below repository root |
| `ModuleNotFoundError` | Dependencies were not installed | Confirm `requirements.txt` is committed and inspect Streamlit build logs |
| Uploaded workbook is empty | Wrong sheet or unsupported structure | Enable all-sheet reading and select the correct file-sheet label |
| Join creates too many rows | Non-unique or many-to-many keys | Audit key uniqueness and aggregate to a common grain before joining |
| No numeric variables available | Numeric columns were imported as text | Inspect decimal separators, symbols and missing-value codes |
| Model fails with singularity | Perfect collinearity, insufficient rows or excessive fixed effects | Reduce predictors, remove duplicates and verify sample rank |
| Cluster method returns one class | Parameters or scaling do not separate observations | Review scaling, DBSCAN `eps`, minimum samples or selected variables |
| Map has unmatched regions | Labels do not match lookup or NUTS level | Use NUTS codes or extend the alias table with verified labels |
| AHP consistency is high | Pairwise judgements conflict | Revisit comparisons; do not report weights mechanically |
| MCDA rank changes sharply | Weight/method sensitivity | Report instability and avoid presenting a unique deterministic winner |
| Publication ZIP is slow | 600-dpi and vector rendering | Allow the calculation to complete; reduce displayed alternatives if required |
| R unavailable | `Rscript` is not installed | Continue with Python or install R separately for replication |

---

## 12. Reproducibility protocol

For every retained result archive:

- release ZIP and `VERSION.txt`;
- original input files and checksums;
- cleaned analytical extract;
- data dictionary and audit workbook;
- research question and pre-specified hypothesis;
- all filters, joins and aggregation rules;
- outcome, predictor and control lists;
- model family, covariance estimator and fixed effects;
- random seed, simulation count and tuning parameters;
- output workbook, HTML report and publication bundle;
- interpretation notes and limitations;
- Python/R versions and installed package versions.

The application provides the necessary exports, but the research team must maintain the project-level audit trail and repository discipline.

---

## 13. Requirements coverage matrix

| Requirement | Implementation | Coverage |
|---|---|---|
| Multiple Excel/CSV files | Simultaneous multi-file upload and all-sheet reading | Full |
| Separate, append or join | Three explicit dataset-relationship modes | Full |
| 1,000 dependent variables | Batch selector and vectorised OLS screen | Full |
| 1,000 independent variables | Batch selector and SVD/pseudoinverse screen | Full |
| OLS and robust econometrics | OLS/WLS/Huber, robust/HAC/clustered covariance, diagnostics | Full |
| Binary, count and fractional outcomes | Logit, probit, Poisson, negative binomial and fractional logit | Full |
| Monte Carlo | OLS uncertainty and stochastic portfolio modules | Full |
| Clustering | Four advanced algorithms, one-variable use, validation and stability | Full |
| Panel analysis | Original R&D panel plus pooled/FE/RE/Hausman laboratory | Full |
| Spatial analysis | Detailed Greece NUTS maps, Moran/LISA and spatial exports | Full |
| Predictive modelling | Seven cross-validated models and permutation importance | Full |
| Time series and multivariate | ADF/KPSS, Granger, ARIMA, PCA, reliability | Full |
| Scenarios and allocation | OLS shock scenario and constrained linear programming | Full with stated policy-design boundary |
| Dedicated MCDA | MAVT, TOPSIS, PROMETHEE II, five weight methods and robustness | Full reusable engine |
| Publication-ready outputs | Exact tables, HTML, colour/B&W PNG 600 dpi, SVG and PDF | Full |
| Copy-ready analytical questions | 52 prompts covering all nineteen modules, with setup and expected output | Full |
| PDF/equation research commands | Evidence scoping, method extraction, replication audit and safe equation/algorithm validation | Full with explicit implementation boundary |
| Explanatory comments | Every table/chart has purpose, reading, meaning, non-claim and next-step guidance in the UI and bundles | Full |
| Python, R and web technologies | Python production, optional R replication, Streamlit/HTML/JS interface | Full |
| Preserve existing functions | Every Version 5.3.2 capability retained | Full |

No new application is required for the forthcoming 2,000-project renewable-energy dataset, provided the supplied file contains stable alternative identifiers and analysable criteria. Dataset-specific recoding, criteria validation, spatial keys and research specifications will still be required when the file arrives.

---

## 14. Limitations and research-governance rules

1. The dashboard cannot make an observational association causal by presentation quality.
2. Automated variable selection can amplify false discovery and specification search.
3. Very wide models can be mathematically estimable through a pseudoinverse yet scientifically unidentified.
4. Regional replication of indicators requires clustered, panel or multilevel reasoning.
5. Missing-data handling changes the effective estimand and must be reported.
6. Maps can visually exaggerate large-area regions and conceal within-region heterogeneity.
7. Internal cluster-validation indices do not establish policy relevance.
8. Predictive accuracy does not establish causal effect or fairness.
9. MCDA weights and value functions are normative modelling decisions, even when derived from data.
10. Scenario and allocation outputs are conditional decision aids, not automatic policy recommendations.
11. Publication bundles require researcher review of titles, captions, units, journal dimensions and accessibility.
12. Confidential or personal data require lawful processing, access controls, retention rules and appropriate hosting.

---

## 15. Version 5.4 interpretation and prompt layer

Module 12B now includes a fifth tab containing 52 copy-ready questions and analytical commands. Each prompt identifies the Research Chair or validated specialist module where it should be used, the setup required and the output to expect. The deterministic interpreter directly supports headline findings, safe/prohibited conclusions, limitations, further research, research-question proposals, methodology/results outlines, reproducible Monte Carlo OLS, PDF-method replication audits and equation/algorithm checks.

Every rendered table and chart now provides a plain-language summary and a detailed explanation. Every Excel export adds an `Output guide` worksheet. Research and publication bundles add the guide in CSV and Markdown, while the Research Chair bundle also contains the complete prompt library. These additions explain evidence; they do not relax causal, identification, measurement or reproducibility safeguards.

---

## 16. Release identification

The expected header is **POSTDOCTORAL ANALYTICAL ENGINE v5.4.0**. The application contains nineteen analytical modules, including **12A. Dedicated MCDA engine** and **12B. Research Command Chair**. The compatibility folder remains `makryvelios_dashboard_v2` and the Streamlit entry point remains `makryvelios_dashboard_v2/app.py`.

The authoritative documentation set consists of this document, `README.md`, `SOURCE_MANIFEST.md`, `CHANGELOG.md`, `GITHUB_STREAMLIT_UPDATE.md` and the focused guides in the `documentation/` directory.
