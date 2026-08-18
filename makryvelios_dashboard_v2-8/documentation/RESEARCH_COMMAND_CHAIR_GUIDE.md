# Research Command Chair — operating guide

## Purpose

Module 12B converts a precisely selected spreadsheet and PDF evidence scope into a reproducible research protocol. It is an additive module: the econometric, GIS, clustering, MCDA, Monte Carlo, predictive and publication modules remain unchanged.

## Cost and privacy

The built-in interpreter is free and requires no API key. Spreadsheet and PDF contents are processed within the running application. A locally installed Ollama model may be selected for richer prose; Ollama is optional and is normally unavailable on Streamlit Community Cloud.

## Guided Chat Autopilot — recommended first route

The first Chair tab is a one-screen substitute for navigating the analytical menu. It automatically uses the active spreadsheet, offers a core paper-question pack, allows any subset or all 52 prepared questions to be loaded, and keeps every question editable in a white field with black type. General questions run without model configuration. Regression, Monte Carlo, prediction, group, time-series and panel requests use the optional Quick model settings expander because the application will not invent outcome, predictor, time or entity roles.

Press **Run selected questions** once. Every question receives one of three explicit verdicts before a response is exported:

- **Feasible — executed:** a validated routine ran and the answer reports computed evidence.
- **Not feasible — statistical/mathematical or logic/setup error:** the request violates theory, contains a logical defect or lacks an indispensable variable mapping.
- **Not feasible — not yet implemented in chat:** the deterministic interpreter cannot execute the method safely; the answer directs the user to the retained specialist menu or to the developers.

The result page supplies genuine prose answers, the feasibility matrix, manuscript-ready methodology/results text, interactive figures and the complete bundle. No instruction-only placeholder is presented as a result.

## Advanced workflow

1. Upload XLSX/CSV files through the existing sidebar and choose the active dataset.
2. Open Module 12B and select no more than 1,000 retained variables.
3. Select an optional year/date variable and exact year interval.
4. Add up to six categorical or numeric row filters.
5. Upload one or more PDFs. Select documents, page intervals and optional keywords.
6. State one principal research question, the algorithm, equation, steps and limitations.
7. Optionally create a derived numeric variable with the safe expression engine.
8. Run the Research Command and inspect every generated table and warning. If no outcome or predictors are chosen, Descriptive profile analyses every numeric variable in the selected scope rather than returning an empty table.
9. Ask questions about the selected evidence. The free interpreter executes a bounded data command first and then describes the computed tables; it does not answer from a generic template.
10. Download the Word/Markdown paper blueprint and the complete reproducibility bundle.

## Copy-ready Prompt Library

Tab 5 contains 52 prompts covering all nineteen dashboard modules. Each entry states where it should be used, the setup required before execution and the output to expect. Prompts include headline findings, safe conclusions, prohibited claims, limitations, further research, research questions and answers, methods/results outlines, OLS/IV/DiD/panel/spatial/time-series/clustering/prediction/MCDA workflows, reproducible Monte Carlo requests, PDF-method extraction and equation/algorithm audits.

Replace square-bracket placeholders with real variable names and settings. A prompt labelled **Research Chair** can be pasted directly into the question box. A prompt labelled with a specialist module should be configured in that module because the app does not silently imitate unsupported estimators.

## Safe equations

The LaTeX equation is preserved for display and manuscript reporting. A separate computable expression can create one derived variable. Allowed components are numeric column names, constants, parentheses, `+`, `-`, `*`, `/`, `**`, `%`, `log`, `log1p`, `exp`, `sqrt` and `abs`. Python statements, attributes, imports, file operations and system commands are rejected.

## Built-in and chat-routed algorithms

- **Descriptive profile:** analytical-sample audit and numeric summaries.
- **Longitudinal trend:** mean, sum, median or count by the selected year and optional group.
- **Correlation screening:** pairwise Spearman associations and p-values.
- **OLS specification:** OLS with HC3 heteroskedasticity-robust standard errors.
- **Custom documented algorithm:** records the supplied steps and limitations, but does not misrepresent prose as validated executable code.

When required variables are mapped, the Guided Chat Autopilot can also execute normality and outlier diagnostics, group comparisons, PCA, Cronbach reliability, advanced clustering, cross-validated prediction, ARIMA forecasting, Monte Carlo OLS and pooled/fixed/random-effects panel comparison. Specialist workflows requiring additional semantic controls—such as instrument/endogenous mappings, treatment/post variables, spatial weights or MCDA value judgements—remain visibly classified rather than guessed.

Use the specialist analytical modules for panel FE/RE, DiD, IV/2SLS, spatial inference, Monte Carlo, clustering, prediction and MCDA after the Research Chair has defined the evidence scope and protocol.

## Natural-language data commands

Questions about missingness, correlations, trends, regression, Monte Carlo, safe conclusions, limitations, further analysis, research questions, methods/results outlines, PDF methods, equation audits or a paper-ready analysis run directly against the saved XLSX/CSV scope. Commands such as `run the analysis as in paper` generate descriptive statistics, a variable-level missingness audit, bounded Spearman screening, longitudinal results when a year variable is selected, HC3 OLS when an outcome and predictors are selected, and method evidence from selected PDF pages. The result tables immediately replace the active Research Chair output and enter the downloadable XLSX and ZIP bundle.

A Monte Carlo request must name a saved outcome/predictor specification and may state the method, repetitions and seed, for example: `Run a wild-bootstrap Monte Carlo OLS for the selected outcome and predictors using 5,000 repetitions and seed 42.` The app exports the coefficient summary, every draw and the settings. Simulation quantifies uncertainty conditional on that model; it does not remove confounding or measurement error.

The app never guesses a dependent variable. If OLS is requested without an explicitly selected outcome and predictor set, it returns the data audit and explains what must be selected before estimation.

For `What is the most striking statistical result?`, the interpreter ranks adequately observed, varying numerical measures after excluding apparent identifiers, codes and time co-ordinates. The answer must contain the leading estimate, n and p-value, an explanation of substantive meaning, the most severe missing-data issue, any invalid outcome/predictor selection, publication-ready wording and the recommended next analysis. The full ranking is exported as `Ranked statistical findings`.

Example questions are displayed immediately above the question box. The supplied R&D data can support requests such as strongest association, missingness risk, change through selected years, explicit OLS interpretation and methods evidenced by selected PDF pages.

## PDF limitations

Text-native PDFs are extracted page by page. Image-only scans require OCR before upload. Keyword matching is literal and does not establish conceptual relevance. Selected passages are research notes, not automatically verified quotations or complete references; always check the original page before submission.

## Paper blueprint

The generated blueprint includes contribution, research question, hypotheses, data scope, algorithm, equation, result tables, interpretation, robustness plan, limitations, recommended manuscript structure and selected PDF locations. It is a structured drafting aid, not an automatically publishable paper.

## Reproducibility bundle

The ZIP contains:

- filtered analytical data in CSV;
- selected PDF evidence in CSV and text;
- protocol JSON;
- every result table in CSV;
- consolidated XLSX workbook;
- paper blueprint in Markdown and Word;
- selected-question answers and feasibility verdicts in Markdown, Word and XLSX/CSV tables;
- a self-contained interactive analytical HTML report;
- standalone Plotly HTML for every generated Chair figure;
- colour and black-and-white 600-dpi PNG plus vector SVG/PDF versions of every generated Chair figure;
- plotted figure data, figure index and figure-by-figure interpretation;
- `Output guide` and `Prompt library` XLSX worksheets;
- interpretation guide in Markdown and CSV;
- complete prompt library in Markdown and CSV;
- a README stating the scientific safeguards and start-here order.

Every guide explains what each output is, why it is used, how to read its signs/intervals/ranks or patterns, what it does not establish and the next defensible step. This commentary is instructional and does not replace subject-matter validation.
