"""Plain-language explanations for tables, charts and downloadable bundles."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _base_entry(name: str, kind: str) -> dict[str, str]:
    label = str(name).replace("_", " ").strip()
    return {
        "output": label,
        "format": kind,
        "what_it_is": "A reproducible analytical output calculated from the selected rows and variables.",
        "why_it_is_used": "It converts the underlying data into evidence that can be checked, compared and reported.",
        "how_to_read_it": "Begin with the title and column labels, confirm the sample size, then examine magnitude and uncertainty before statistical significance.",
        "what_the_pattern_means": "Larger or smaller values matter only in relation to the variable definition, unit and research question.",
        "what_it_does_not_mean": "The output does not by itself prove causality, remove bias, validate measurement or guarantee practical importance.",
        "recommended_next_step": "Check definitions, missingness, sample restrictions and robustness before using the output in a manuscript.",
        "plain_language_summary": "Read this output as evidence conditional on the selected data and settings, not as an automatic conclusion.",
    }


def explain_output(name: str, table: pd.DataFrame | None = None, kind: str = "table") -> dict[str, str]:
    """Return a detailed, non-technical guide tailored to an output label."""
    entry = _base_entry(name, kind)
    key = str(name).casefold().replace("_", " ")
    table = table if isinstance(table, pd.DataFrame) else pd.DataFrame()

    if "scope" in key or "filtered data" in key:
        entry.update(
            what_it_is="The exact analytical sample retained after the selected files, columns, years and row filters were applied.",
            why_it_is_used="It makes the denominator and evidence boundary explicit, so every later result can be reproduced.",
            how_to_read_it="Check the number of records and variables first. Confirm that the unit of observation is one project, region-year or other intended unit.",
            what_the_pattern_means="A smaller scope may be correct if deliberate filters were applied; an unexpected loss of rows indicates a filtering or join problem.",
            what_it_does_not_mean="A large sample is not automatically representative, unbiased or suitable for causal inference.",
            recommended_next_step="Verify inclusion/exclusion rules, duplicates, identifiers and time coverage before modelling.",
            plain_language_summary="This tells you exactly which observations the subsequent calculations describe.",
        )
    elif "descriptive" in key or "summary" in key and "simulation" not in key:
        entry.update(
            what_it_is="A numerical portrait of each selected variable: usable observations, mean, spread, minimum, quartiles, median, maximum and missingness.",
            why_it_is_used="It reveals scale, skew, unusual ranges and whether enough data exist before formal modelling.",
            how_to_read_it="Compare the mean with the median: a large difference suggests skew. Compare the 25th and 75th percentiles to understand the middle half of observations. Check minimum and maximum for implausible values.",
            what_the_pattern_means="A higher mean means a higher average in the variable's own units; a larger standard deviation means observations are more dispersed.",
            what_it_does_not_mean="Descriptive differences are not tests of significance and cannot identify why values differ.",
            recommended_next_step="Plot distributions, inspect outliers and choose a test or model suited to the variable type and research question.",
            plain_language_summary="This table answers: what values are typical, how varied are they, and are any values or missingness concerning?",
        )
    elif "missing" in key or "quality" in key or "audit" in key or "outlier" in key:
        entry.update(
            what_it_is="A data-quality diagnostic showing incomplete, duplicated, unusual or potentially invalid observations.",
            why_it_is_used="Poor data quality can change the effective sample, bias estimates and create misleading significance.",
            how_to_read_it="Prioritise variables with the highest missing percentage and observations flagged by robust IQR/MAD rules. Determine whether missingness is structural, random or caused by coding.",
            what_the_pattern_means="High missingness means fewer usable cases; many outliers may reflect genuine heterogeneity or data errors.",
            what_it_does_not_mean="A flagged value is not automatically wrong, and missing values should not be deleted mechanically.",
            recommended_next_step="Check source records, document exclusions and compare complete-case results with defensible alternative missing-data treatments.",
            plain_language_summary="This identifies where the data may be too incomplete or unusual to support a reliable conclusion.",
        )
    elif "correlation" in key or "association" in key:
        entry.update(
            what_it_is="A measure of whether two variables tend to move together. Spearman correlation assesses monotonic ordering and ranges from −1 to +1.",
            why_it_is_used="It screens relationships, identifies redundant measures and helps detect multicollinearity before regression.",
            how_to_read_it="Values near +1 indicate a strong same-direction relationship; values near −1 indicate a strong inverse relationship; values near 0 indicate little monotonic relationship. Read n and the adjusted p/q-value with the coefficient.",
            what_the_pattern_means="Positive means higher values of one variable generally accompany higher values of the other. Negative means higher values accompany lower values. Near zero means no clear monotonic pattern.",
            what_it_does_not_mean="Correlation does not prove causality, temporal order or absence of a non-linear relationship.",
            recommended_next_step="Inspect a scatterplot, verify definitions, control multiple testing and use a theory-led model if an adjusted relationship is required.",
            plain_language_summary="This shows whether two measures move together, in opposite directions or show no clear ordered relationship.",
        )
    elif "coefficient" in key or "regression" in key or "ols" in key or "2sls" in key or "difference" in key:
        entry.update(
            what_it_is="A model table estimating how the outcome changes with each predictor while the other included predictors are held constant.",
            why_it_is_used="It quantifies conditional relationships and their uncertainty under an explicit equation.",
            how_to_read_it="The coefficient gives direction and size. A positive coefficient means a higher predicted outcome; a negative coefficient means a lower predicted outcome. A 95% confidence interval crossing zero is not conventionally distinguishable from no conditional association at the 5% level.",
            what_the_pattern_means="Magnitude must be translated into the outcome and predictor units. Robust standard errors protect inference against specified variance problems, not omitted-variable bias.",
            what_it_does_not_mean="A significant coefficient is not necessarily causal, practically important or correctly specified.",
            recommended_next_step="Inspect fit, residuals, VIF, influential observations and theoretically justified robustness specifications.",
            plain_language_summary="This estimates the direction and size of an adjusted relationship, together with how uncertain that estimate is.",
        )
    elif "fit" in key or "diagnostic" in key or "residual" in key or "vif" in key or "hausman" in key:
        entry.update(
            what_it_is="A model-checking output describing explanatory/predictive fit, assumption problems or the relative suitability of competing specifications.",
            why_it_is_used="A coefficient table is unsafe to interpret without checking whether the model behaves adequately.",
            how_to_read_it="Use R² only as the share of sample variation fitted by the model. Lower AIC/BIC is preferable only among models fitted to the same outcome and sample. Large VIF values indicate overlapping predictors. Inspect diagnostic p-values alongside plots and substantive plausibility.",
            what_the_pattern_means="Better fit means closer in-sample or validated predictions, not necessarily better causal identification.",
            what_it_does_not_mean="Good fit cannot rule out confounding, reverse causality, leakage or measurement error.",
            recommended_next_step="Compare validated alternatives, inspect residual plots and report all specification changes transparently.",
            plain_language_summary="This checks whether the statistical model is behaving well enough to interpret responsibly.",
        )
    elif "monte carlo" in key or "simulation" in key or "draw" in key or "uncertainty" in key:
        entry.update(
            what_it_is="Repeated simulated or resampled estimates generated under a fixed seed and stated uncertainty mechanism.",
            why_it_is_used="It shows how much a result could vary across repeated plausible samples or uncertain scenarios.",
            how_to_read_it="Focus on the simulation median/mean, interval, probability of being positive or negative, and the share of draws crossing zero. A narrow distribution indicates greater stability under the simulated assumptions.",
            what_the_pattern_means="If nearly all draws lie on one side of zero, the sign is stable under that simulation design. A wide or two-sided distribution indicates substantial uncertainty.",
            what_it_does_not_mean="Simulation does not repair biased source data or justify unrealistic assumptions; results are conditional on the chosen mechanism and seed.",
            recommended_next_step="Repeat with alternative defensible simulation methods, seeds and uncertainty parameters and compare conclusions.",
            plain_language_summary="This asks whether the conclusion remains similar when the analysis is repeated under explicitly modelled uncertainty.",
        )
    elif "longitudinal" in key or "trend" in key or "time series" in key or "forecast" in key or "granger" in key:
        entry.update(
            what_it_is="A time-ordered summary or model showing how a measure changes across years or periods.",
            why_it_is_used="It identifies direction, breaks, persistence, seasonality and possible temporal ordering.",
            how_to_read_it="Check the first and last periods, intervening reversals, missing years and whether values are totals, means or medians. A rising line means higher recorded values over time; a falling line means lower values.",
            what_the_pattern_means="A trend describes temporal movement. Forecast intervals widen as uncertainty increases.",
            what_it_does_not_mean="A trend does not prove that time caused the change; Granger precedence is not philosophical causality.",
            recommended_next_step="Test structural breaks, stationarity and alternative aggregations; consider panel models when repeated regions/entities exist.",
            plain_language_summary="This shows what changed over time, when it changed and how stable that movement appears.",
        )
    elif "panel" in key or "fixed effect" in key or "random effect" in key:
        entry.update(
            what_it_is="A longitudinal model using repeated observations for the same regions, organisations or other entities.",
            why_it_is_used="It separates within-entity change over time from stable differences between entities.",
            how_to_read_it="A fixed-effects coefficient describes how the outcome changes when a predictor changes within the same entity. Compare pooled, fixed and random-effects results and inspect the Hausman diagnostic.",
            what_the_pattern_means="Consistent signs across specifications strengthen robustness; large differences indicate sensitivity to unobserved entity characteristics.",
            what_it_does_not_mean="Fixed effects do not remove time-varying confounding, measurement error or reverse causality.",
            recommended_next_step="Check within-entity variation, clustered errors, time effects, lags and alternative outcome distributions.",
            plain_language_summary="This asks whether change within the same region or organisation is associated with change in the outcome.",
        )
    elif "moran" in key or "lisa" in key or "spatial" in key or "map" in key or "geograph" in key:
        entry.update(
            what_it_is="A geographical display or spatial-dependence diagnostic linking values to Greek regions or spatial units.",
            why_it_is_used="It shows where outcomes are concentrated and whether neighbouring areas have unusually similar or dissimilar values.",
            how_to_read_it="On a choropleth, darker/lighter areas represent the legend values. Positive Moran’s I indicates clustering of similar values; negative values indicate neighbouring dissimilarity; LISA labels local hot spots, cold spots and spatial outliers.",
            what_the_pattern_means="Spatial concentration may reveal regional systems, shared conditions or spillovers that deserve investigation.",
            what_it_does_not_mean="A map does not control for population, project counts, scale or spatial confounding unless the measure/model explicitly does so.",
            recommended_next_step="Check rates versus totals, alternative spatial weights, permutation significance and spatial regression where justified.",
            plain_language_summary="This shows where the phenomenon is concentrated and whether neighbouring regions form meaningful spatial patterns.",
        )
    elif "cluster" in key or "segment" in key or "typolog" in key:
        entry.update(
            what_it_is="An unsupervised grouping of observations that are similar on the selected standardised variables.",
            why_it_is_used="It discovers empirical project or regional typologies without pre-assigning categories.",
            how_to_read_it="Use cluster profiles to see which variables are above or below the sample average. Silhouette/stability values indicate whether groups are separated and reproducible. Cluster numbers are labels, not rankings.",
            what_the_pattern_means="Distinct profiles indicate different combinations of characteristics; they do not mean one group is inherently better.",
            what_it_does_not_mean="Clusters are not causal classes and can change with variable selection, scaling, method, k or random seed.",
            recommended_next_step="Test alternative algorithms/k, stability and external validity, then give clusters substantive names only after inspecting profiles.",
            plain_language_summary="This groups similar projects or regions and explains what distinguishes each group.",
        )
    elif "pca" in key or "component" in key:
        entry.update(
            what_it_is="A dimension-reduction result combining correlated variables into a smaller set of components.",
            why_it_is_used="It simplifies high-dimensional data and can reveal shared latent structure.",
            how_to_read_it="Explained variance shows how much information each component retains; loadings show which original variables define it and in which direction.",
            what_the_pattern_means="Large loadings identify the variables most responsible for a component; signs are relative and may be reversed without changing the solution.",
            what_it_does_not_mean="A component is not automatically a validated theoretical construct.",
            recommended_next_step="Check adequacy, stability and interpretability; validate any proposed scale with reliability and theory.",
            plain_language_summary="This condenses many overlapping variables into a few summary dimensions.",
        )
    elif "predict" in key or "importance" in key or "cross validated" in key or "model performance" in key:
        entry.update(
            what_it_is="Out-of-sample predictive evidence comparing algorithms or showing which variables improve prediction.",
            why_it_is_used="It evaluates whether the model generalises beyond the observations used for fitting.",
            how_to_read_it="Lower cross-validated RMSE/MAE is better; higher cross-validated R² is better. Permutation importance shows the performance loss when a variable is disrupted.",
            what_the_pattern_means="A better model predicts held-out observations more accurately under the validation design.",
            what_it_does_not_mean="Predictive importance is not a causal effect and may be shared among correlated predictors.",
            recommended_next_step="Use nested/repeated validation where feasible, inspect calibration and test on a genuinely external dataset.",
            plain_language_summary="This tests which model predicts unseen cases best and which variables contribute to that prediction.",
        )
    elif "mcda" in key or "ranking" in key or "weight" in key or "acceptability" in key or "sensitivity" in key:
        entry.update(
            what_it_is="A multi-criteria decision result combining several benefits, costs or risks into an auditable ranking.",
            why_it_is_used="It supports transparent prioritisation when no single outcome captures the whole decision.",
            how_to_read_it="Check criterion directions and weights first, then scores/ranks. Sensitivity and rank-acceptability results show whether rankings survive plausible weight uncertainty.",
            what_the_pattern_means="A higher score or better rank means stronger performance under the chosen method, criteria, directions and weights.",
            what_it_does_not_mean="The ranking is not objective truth; it formalises the stated value judgements and data quality.",
            recommended_next_step="Compare methods, vary weights, review AHP consistency and document stakeholder justification for every criterion.",
            plain_language_summary="This combines several competing considerations and tests whether the resulting priority order is robust.",
        )
    elif "normality" in key:
        entry.update(
            what_it_is="A diagnostic of whether a variable resembles a normal bell-shaped distribution.",
            why_it_is_used="Some classical procedures use normality assumptions for residuals or small-sample inference.",
            how_to_read_it="A small p-value indicates detectable departure from normality, but large datasets detect trivial departures. Inspect histograms/Q–Q plots and use robust or distribution-appropriate methods.",
            what_the_pattern_means="Non-normality may reflect skew, heavy tails, mixtures, censoring or outliers.",
            what_it_does_not_mean="A non-normal raw variable does not automatically invalidate regression; residual behaviour and estimator assumptions matter.",
            recommended_next_step="Inspect the distribution, consider transformations or GLMs and use robust inference where appropriate.",
            plain_language_summary="This checks whether the distribution has a bell shape, but the p-value must not be interpreted mechanically.",
        )
    elif "frequency" in key or "categorical" in key or "chi" in key:
        entry.update(
            what_it_is="Counts, percentages or an association test for categorical groups.",
            why_it_is_used="It describes composition and tests whether category distributions differ more than expected by chance.",
            how_to_read_it="Compare counts and column/row percentages, not counts alone. For chi-square, inspect the p-value, expected-count warnings and an effect-size measure such as Cramér’s V.",
            what_the_pattern_means="Different percentages indicate group composition differences; effect size shows whether the difference is substantively small or large.",
            what_it_does_not_mean="A significant chi-square test does not identify causality or which cell alone caused the overall association.",
            recommended_next_step="Inspect standardised residuals, combine sparse categories only with justification and model adjusted probabilities if needed.",
            plain_language_summary="This shows how observations are distributed across categories and whether two classifications are associated.",
        )
    elif "method" in key or "pdf" in key or "evidence" in key:
        entry.update(
            what_it_is="An index of method-related passages found in the selected PDF documents and page ranges.",
            why_it_is_used="It links analytical choices to documentary evidence without silently inventing a methodology.",
            how_to_read_it="Use the document and page columns to return to the original source. Treat snippets as navigation aids, not complete quotations.",
            what_the_pattern_means="A detected method indicates textual mention, not proof that every implementation detail has been recovered.",
            what_it_does_not_mean="The app cannot guarantee bibliographic accuracy, conceptual equivalence or exact replication from a keyword hit alone.",
            recommended_next_step="Verify the full methods section, map every variable and assumption, then run only methods supported by the available data and implemented engine.",
            plain_language_summary="This shows where the uploaded documents discuss analytical methods and what must be checked before replication.",
        )

    if not table.empty:
        entry["plain_language_summary"] += f" The displayed table contains {len(table):,} row(s) and {table.shape[1]:,} column(s)."
        if "missing_percent" in table:
            values = pd.to_numeric(table["missing_percent"], errors="coerce")
            if values.notna().any():
                index = values.idxmax()
                variable = table.loc[index, "variable"] if "variable" in table else "the leading variable"
                entry["plain_language_summary"] += f" Highest missingness: {variable} ({values.loc[index]:.2f}%)."
        rho_column = next((c for c in ["spearman_rho", "rho", "correlation"] if c in table), None)
        if rho_column:
            rho = pd.to_numeric(table[rho_column], errors="coerce")
            if rho.notna().any():
                index = rho.abs().idxmax()
                entry["plain_language_summary"] += f" Strongest displayed association: {rho.loc[index]:.3f}."
    return entry


def build_output_guide(
    tables: dict[str, pd.DataFrame] | None = None,
    charts: list[str] | None = None,
) -> pd.DataFrame:
    rows = [explain_output(name, table, "table") for name, table in (tables or {}).items() if "output guide" not in name.casefold()]
    rows.extend(explain_output(name, None, "chart") for name in (charts or []))
    return pd.DataFrame(rows)


def output_guide_markdown(guide: pd.DataFrame) -> str:
    if guide.empty:
        return "# Output interpretation guide\n\nNo outputs were generated."
    sections = ["# Output interpretation guide", "", "Use this guide together with the exact table/chart and the declared analytical protocol."]
    labels = [
        ("what_it_is", "What it is"),
        ("why_it_is_used", "Why it is used"),
        ("how_to_read_it", "How to read it"),
        ("what_the_pattern_means", "What the pattern means"),
        ("what_it_does_not_mean", "What it does not mean"),
        ("recommended_next_step", "What to do next"),
    ]
    for row in guide.itertuples(index=False):
        sections.extend(["", f"## {row.output} ({row.format})", "", row.plain_language_summary])
        for field, label in labels:
            sections.extend(["", f"**{label}.** {getattr(row, field)}"])
    return "\n".join(sections) + "\n"

