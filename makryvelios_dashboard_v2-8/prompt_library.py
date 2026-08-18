"""Copy-ready research prompts mapped to the dashboard's analytical modules."""
from __future__ import annotations

import pandas as pd


PROMPTS = [
    ("Orientation", "1. Executive overview", "Research Chair", "What are the five most important facts about the selected dataset, and which one deserves immediate investigation?", "An active XLSX/CSV scope", "Ranked findings, data-quality warnings and next analyses"),
    ("Orientation", "1. Executive overview", "Research Chair", "What is the most striking statistical result, why is it striking, and how should I report it cautiously?", "An active XLSX/CSV scope", "Magnitude, n, adjusted significance, limitation and publication wording"),
    ("Data quality", "2. Data hub & audit", "Research Chair", "Which variables have the greatest missingness, which are still usable, and which should be excluded from modelling?", "Selected variables", "Missingness ranking and modelling implications"),
    ("Data quality", "2. Data hub & audit", "Module 2", "Check duplicates, identifier uniqueness, impossible values, inconsistent years and extreme outliers. Explain every problem and do not delete anything automatically.", "Active dataset", "Audit tables and recommended corrections"),
    ("Research design", "3. Research questions", "Research Chair", "Propose ten original research questions that this dataset can answer, identify the outcome and predictors for each, and classify each as descriptive, associative, predictive or potentially causal.", "Selected columns and definitions", "Research-question catalogue with feasible methods"),
    ("Research design", "3. Research questions", "Research Chair", "Convert my principal research question into testable hypotheses, define each variable and state the expected direction without inventing causal identification.", "A written principal research question", "Hypotheses and operationalisation"),
    ("Description", "4. Descriptive statistics", "Module 4", "Describe all selected numeric variables, identify skew, unusual ranges and variables for which the median is more informative than the mean.", "Numeric variables", "Descriptive statistics and distribution commentary"),
    ("Description", "4. Descriptive statistics", "Module 4", "Summarise the composition of projects by region, sector, beneficiary type and scientific field using counts and percentages.", "Categorical variables", "Frequency tables and composition charts"),
    ("Association", "4. Descriptive statistics", "Research Chair", "Which two substantive indicators have the strongest positive association, strongest inverse association and weakest association? Exclude identifiers and apply multiple-testing correction.", "At least three numeric measures", "Ranked correlations with safeguards"),
    ("Hypothesis testing", "5. Hypothesis tests", "Module 5", "Test whether [OUTCOME] differs across [GROUP]. Select the appropriate test, report assumptions, effect size, confidence interval and corrected pairwise comparisons.", "Outcome and group variables", "Test table, effect size and interpretation"),
    ("Hypothesis testing", "5. Hypothesis tests", "Module 5", "Test whether [CATEGORY_A] is associated with [CATEGORY_B]. Report expected-count problems, Cramér’s V and which cells drive the association.", "Two categorical variables", "Chi-square evidence and effect size"),
    ("OLS", "6. OLS & econometric laboratory", "Research Chair / Module 6", "Estimate [OUTCOME] on [PREDICTOR_1], [PREDICTOR_2] and [CONTROL] using OLS with HC3 errors. Explain every coefficient in the original units and state what cannot be interpreted causally.", "Explicit outcome and predictors", "Coefficients, intervals, fit and diagnostics"),
    ("OLS", "6. OLS & econometric laboratory", "Module 6", "Check heteroskedasticity, residual behaviour, influential observations and multicollinearity for the selected OLS model. Explain what each diagnostic changes in the paper.", "An estimated OLS model", "Diagnostics, VIF and remedies"),
    ("Identification", "6. OLS & econometric laboratory", "Module 6", "Assess whether [INSTRUMENT] is a defensible instrument for [ENDOGENOUS_VARIABLE] when estimating its relationship with [OUTCOME]. Report first-stage strength and all exclusion-restriction limitations.", "Outcome, endogenous variable and instrument", "2SLS estimates and instrument diagnostics"),
    ("Identification", "6. OLS & econometric laboratory", "Module 6", "Estimate a difference-in-differences model for [OUTCOME] using [TREATMENT], [POST] and their interaction. Explain the interaction and the parallel-trends requirement.", "Outcome, treatment and post indicators", "DiD coefficient and identification caveats"),
    ("Monte Carlo", "6A. Monte Carlo & uncertainty", "Research Chair / Module 6A", "Run a wild-bootstrap Monte Carlo analysis for the selected OLS model using 5,000 replications and seed 42. Report the simulated coefficient distribution, 95% interval and probability that each effect is positive.", "Explicit outcome and predictors", "Reproducible simulation summary and draws"),
    ("Monte Carlo", "6A. Monte Carlo & uncertainty", "Module 6A", "Run residual-bootstrap and parametric-normal simulations with 5,000 replications and seed 42, compare them with wild bootstrap, and explain whether the substantive conclusion changes.", "An OLS specification", "Cross-method uncertainty comparison"),
    ("Portfolio uncertainty", "6A. Monte Carlo & uncertainty", "Module 6A", "Simulate project selection under uncertain cost and benefit using [BUDGET], cost CV [VALUE], benefit CV [VALUE], 10,000 draws and seed 42. Identify robust selections and downside risk.", "Cost, benefit, project ID and budget", "Selection probabilities and risk distribution"),
    ("High-dimensional", "7. 1,000 × 1,000 batch engine", "Module 7", "Screen all selected outcomes against all selected predictors, control the false-discovery rate, rank stable associations and flag models with inadequate observations or singularity.", "Multiple outcomes and predictors", "Batch coefficients, fit and adjusted significance"),
    ("Regional R&D", "8. Original R&D regional panel", "Module 8", "Reproduce the relevant EE1–EE9 regional specification, state the exact unit of analysis and compare the result with the documented Stata model.", "R&D-compatible dataset", "Recovered model and reproducibility comparison"),
    ("Panel models", "8A. Panel model laboratory", "Module 8A", "Estimate pooled OLS, two-way fixed effects and random effects for [OUTCOME] on [PREDICTORS] by [REGION] and [YEAR]. Compare signs, magnitudes and the Hausman test.", "Entity, year, outcome and varying predictors", "Panel coefficient and fit comparison"),
    ("Panel models", "8A. Panel model laboratory", "Module 8A", "What can the fixed-effects model identify that cross-sectional OLS cannot, and which time-varying confounders may still bias the estimate?", "Estimated panel suite", "Identification explanation and limitations"),
    ("Spatial", "9. Detailed Greece GIS", "Module 9", "Map [MEASURE] across Greek NUTS-2 regions using totals and rates separately. Explain which map is substantively appropriate and why.", "Region and measure", "Colour/B&W maps and scale warning"),
    ("Spatial", "9. Detailed Greece GIS", "Module 9", "Test global Moran’s I and local LISA for [MEASURE] with 999 permutations. Identify clusters and spatial outliers without claiming spatial causality.", "Regional measure", "Spatial diagnostics and interpretation"),
    ("Time series", "10. Time series & multivariate", "Module 10", "Plot [MEASURE] through [YEAR], test stationarity, identify structural breaks and state whether trend modelling is defensible.", "Year and numeric measure", "Trend, tests and limitations"),
    ("Time series", "10. Time series & multivariate", "Module 10", "Forecast [MEASURE] for [HORIZON] periods using an appropriate ARIMA specification. Report forecast intervals and explain why they widen.", "Ordered time series", "Forecast table and plot"),
    ("Multivariate", "10. Time series & multivariate", "Module 10", "Run PCA on the selected indicators. Explain retained variance, the variables defining each component and whether any component has a defensible substantive label.", "Several numeric indicators", "Loadings, scores and variance"),
    ("Measurement", "10. Time series & multivariate", "Module 10", "Calculate Cronbach’s alpha for [ITEMS]. Explain whether a combined scale is defensible and whether any item appears redundant.", "Conceptually related items", "Reliability table and item assessment"),
    ("Clustering", "10A. Advanced clustering & segmentation", "Module 10A", "Cluster projects using [VARIABLES], choose k automatically up to 8 with seed 42, and explain each cluster in plain language without treating cluster numbers as rankings.", "At least one numeric variable", "Assignments, profiles, diagnostics and projection"),
    ("Clustering", "10A. Advanced clustering & segmentation", "Module 10A", "Create clusters based only on resource absorption. Then repeat with budget, duration, employment and innovation indicators and explain how the typology changes.", "Absorption and optional additional measures", "One-dimensional and multivariate comparison"),
    ("Prediction", "10B. Predictive model laboratory", "Module 10B", "Compare OLS, Ridge, Lasso, Elastic Net, random forest, extra trees and gradient boosting for predicting [OUTCOME] using five-fold cross-validation and seed 42.", "Continuous outcome and predictors", "Out-of-sample performance ranking"),
    ("Prediction", "10B. Predictive model laboratory", "Module 10B", "Which variables contribute most to prediction of [OUTCOME], and why must predictive importance not be described as a causal effect?", "Completed predictive comparison", "Permutation importance and causal warning"),
    ("Publication", "11. Publication figures & HTML report", "Module 11", "Create a publication-ready [CHART TYPE] of [Y] by [X], with colour and black-and-white versions, 600-dpi PNG, SVG, PDF, plotted data and a complete figure interpretation.", "Chosen x/y and optional group", "Publication bundle with guide"),
    ("Publication", "11. Publication figures & HTML report", "Module 11", "Prepare a self-contained HTML report containing the selected tables, figures, methods, limitations and reproducibility settings.", "Completed analyses", "Portable report"),
    ("Scenario", "12. Scenario & allocation engine", "Module 12", "Allocate a total budget of [AMOUNT] across projects to maximise [BENEFIT] subject to [CONSTRAINTS]. Explain the objective, constraints, shadow trade-offs and infeasibility risks.", "Cost, benefit and constraints", "Optimal allocation and scenario table"),
    ("MCDA", "12A. Dedicated MCDA engine", "Module 12A", "Rank projects using [CRITERIA], with each criterion marked maximise or minimise. Compare MAVT, TOPSIS and PROMETHEE II and explain disagreements.", "Alternatives and criteria", "Method rankings and agreement"),
    ("MCDA", "12A. Dedicated MCDA engine", "Module 12A", "Use AHP weights from my pairwise judgements, report the consistency ratio, run 10,000 Monte Carlo weight draws with seed 42 and identify alternatives robustly ranked first.", "Criteria and pairwise judgements", "Weights, sensitivity and rank acceptability"),
    ("PDF methodology", "12B. Research Command Chair", "Research Chair", "Extract the methodology from the selected PDF pages. List the sample, variables, transformations, equation, estimator, covariance rule, diagnostics and robustness checks with page references.", "Selected PDF pages", "Auditable methodology extraction"),
    ("PDF methodology", "12B. Research Command Chair", "Research Chair", "Compare the paper’s methodology with the available XLSX/CSV variables. State exactly what can be replicated, what requires recoding and what cannot be executed with the available evidence.", "Selected PDF pages and spreadsheet scope", "Replication feasibility matrix"),
    ("PDF methodology", "12B. Research Command Chair", "Research Chair", "Replicate only the supported parts of the selected paper’s methodology using the current data. Do not invent missing variables, assumptions or equations; document every deviation.", "Selected PDF pages and mapped variables", "Supported execution and deviation log"),
    ("Equations", "12B. Research Command Chair", "Research Chair", "Check this equation against statistical and mathematical theory: [EQUATION]. Define every symbol, identify estimand and assumptions, detect invalid operations and state whether the available variables can implement it.", "Equation and variable definitions", "Equation audit and implementation plan"),
    ("Algorithms", "12B. Research Command Chair", "Research Chair", "Audit this algorithm step by step for mathematical correctness, data leakage, circular reasoning, unsupported assumptions and reproducibility: [PASTE ALGORITHM].", "Algorithm steps", "Algorithm validation checklist"),
    ("Algorithms", "12B. Research Command Chair", "Research Chair", "Execute this supported algorithm on the selected scope using seed [SEED] and [REPETITIONS] repetitions: [PASTE ALGORITHM]. Return the exact settings, tables, diagnostics and limitations.", "Implemented algorithm and mapped variables", "Execution or explicit unsupported-step report"),
    ("Conclusions", "12B. Research Command Chair", "Research Chair", "What can I safely conclude from the current results? Separate descriptive, associative, predictive and causal statements.", "Completed results", "Permitted-claim matrix"),
    ("Conclusions", "12B. Research Command Chair", "Research Chair", "What can I not conclude from these results, even if some p-values are below 0.05?", "Completed results", "Prohibited claims and reasons"),
    ("Limitations", "12B. Research Command Chair", "Research Chair", "Write a limitations section based only on the sample, missingness, measurement, model assumptions, time coverage, spatial coverage and identification strategy shown in the outputs.", "Completed scope and results", "Paper-ready limitations"),
    ("Further analysis", "12B. Research Command Chair", "Research Chair", "Which additional analyses are justified by the current findings? Rank them by scientific value, feasibility and the extra variables required.", "Completed results", "Prioritised analytical roadmap"),
    ("Further research", "12B. Research Command Chair", "Research Chair", "Propose a future-research agenda that follows logically from these findings, distinguishing extensions possible with current data from those requiring new data.", "Completed results", "Near-term and longer-term agenda"),
    ("Paper methodology", "12B. Research Command Chair", "Research Chair", "Give me a section-by-section outline of the methodology actually used, including sample construction, variable operationalisation, equations, estimators, diagnostics, robustness and reproducibility settings.", "Executed protocol", "Methods-section outline"),
    ("Paper results", "12B. Research Command Chair", "Research Chair", "Give me a section-by-section outline of the results. Order descriptive evidence, primary estimates, diagnostics, robustness and secondary findings without overstating causality.", "Completed results", "Results-section outline"),
    ("Paper development", "12B. Research Command Chair", "Research Chair", "Generate research questions and concise evidence-based answers from the current results. For each, state the method, numerical evidence, safe conclusion and remaining uncertainty.", "Completed results", "Question-and-answer results catalogue"),
    ("Reproducibility", "13. Methods & reproducibility", "Module 13", "Create a reproducibility checklist containing data version, filters, transformations, equations, estimators, covariance rules, seeds, repetitions, software versions and exported files.", "Completed workflow", "Audit checklist"),
]


def prompt_library() -> pd.DataFrame:
    columns = ["category", "module", "where_to_use", "copy_ready_prompt", "required_setup", "expected_output"]
    return pd.DataFrame(PROMPTS, columns=columns)


def prompt_library_markdown(frame: pd.DataFrame | None = None) -> str:
    data = prompt_library() if frame is None else frame
    lines = [
        "# Makryvelios Research Analytics — Copy-ready Prompt Library",
        "",
        "Replace square-bracket placeholders before use. ‘Research Chair’ prompts can be pasted into Module 12B. ‘Module’ prompts describe the settings/actions to reproduce in the named specialist module; the free interpreter does not silently execute an unsupported estimator.",
    ]
    for (category, module), group in data.groupby(["category", "module"], sort=False):
        lines.extend(["", f"## {category} — {module}"])
        for row in group.itertuples(index=False):
            lines.extend([
                "",
                f"**Where to use:** {row.where_to_use}",
                "",
                "```text",
                row.copy_ready_prompt,
                "```",
                f"Required setup: {row.required_setup}",
                f"Expected output: {row.expected_output}",
            ])
    return "\n".join(lines) + "\n"

