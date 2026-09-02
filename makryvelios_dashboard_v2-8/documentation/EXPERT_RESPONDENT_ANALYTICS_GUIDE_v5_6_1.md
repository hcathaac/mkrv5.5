# Expert Respondent Analytics and Empirical ITA Weights — v5.6.1

## Purpose

Module **12A.2** preserves the complete respondent-level evidence instead of reducing the expert panel to one mean coefficient. It is schema-agnostic: the exact respondent identifier, ordered C1-Cn preference fields and optional subgroup variable are mapped after the expert dataset is selected.

The module is fully operational before the two expert papers arrive. It does not predict their conclusions or silently impose a confirmatory model. Once the papers are supplied, their final constructs, hypotheses and specifications must be checked against the mapped fields and the relevant analyses rerun.

## Data handling

Raw responses are retained in exports. For ITA use, every valid respondent vector is converted to relative non-negative weights that sum to one. Complete-case analysis is the default. Criterion-median imputation is available but must be justified and reported. Rows with negative values, incomplete values under complete-case handling, or a zero total are excluded and counted.

Normalising Likert importance ratings creates relative priorities; it does not demonstrate that the items form a unidimensional psychometric scale. Correlations among normalised weights are compositional and may be negative partly because the vector has a fixed sum.

## Outputs

- Criterion means, medians, dispersion, range and reproducible bootstrap confidence intervals.
- Full violin/box distributions and respondent-by-criterion heatmap.
- Kendall's W for concordance in criterion ordering.
- Spearman dependence matrix.
- PCA explained variance and loadings.
- Exploratory K-means preference segments, selected by the strongest silhouette score over feasible k=2–6, with segment profiles.
- Optional Kruskal-Wallis subgroup tests, Benjamini-Hochberg adjusted p-values and epsilon-squared effect sizes.
- Complete respondent, diagnostics, XLSX, CSV and JSON evidence exports.

## Empirical Hybrid ITA-RW bridge

The user may activate the complete normalised respondent matrix for the next Hybrid ITA-RW run. The project ITA module verifies that the empirical matrix has the same number and order of criteria as the project C1-Cn mapping.

In empirical mode, each ITA round samples complete respondent vectors rather than independently sampling criterion marginals. This preserves within-respondent preference structure. The vectors progressively contract towards their empirical centre as the rounds advance; the final round converges to that centre. Score uncertainty remains independently controlled by the ITA settings. The export records the weight source, respondent count, random seed and all thresholds.

## Publication boundary

Segmentation and PCA are exploratory. Subgroup associations are not causal. Kendall's W measures agreement, not correctness. The two forthcoming papers remain authoritative for the final constructs, hypotheses, covariates, estimators and manuscript interpretation.
