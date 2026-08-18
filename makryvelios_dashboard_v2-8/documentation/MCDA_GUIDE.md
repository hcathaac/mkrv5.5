# Dedicated MCDA Engine - Method and Operating Guide

Version 5.2.1 provides a reusable multi-criteria decision-analysis engine for renewable-energy projects, R&D investments, regions, municipal programmes and other alternatives.

## Inputs

- An optional unique alternative identifier.
- Between two and fifty numeric criteria.
- A Maximise or Minimise direction for every criterion.
- Median imputation or complete-case handling.
- Equal, user-defined, Entropy, CRITIC or AHP weights.
- MAVT, TOPSIS and/or PROMETHEE II ranking.
- A primary MAVT/TOPSIS method for sensitivity and Monte Carlo analysis.
- Weight-perturbation range, Dirichlet concentration, number of draws and seed.

## AHP limit

AHP is limited to fifteen criteria because pairwise judgements increase as m(m - 1)/2. The matrix uses the upper triangle as authoritative and reconstructs reciprocal comparisons. A consistency ratio above approximately 0.10 is a warning that judgements should be reviewed.

## Interpretation

MAVT is an additive, compensatory value model. TOPSIS ranks closeness to a weighted ideal. PROMETHEE II uses positive, negative and net preference flows. Divergent ranks are substantive evidence of method sensitivity.

Entropy and CRITIC weights are data-responsive, not neutral policy preferences. Monte Carlo rank probabilities quantify robustness to the specified weight distribution, not the probability of project implementation or success.

## Outputs

- Method-specific scores and ranks.
- Consensus rank when multiple methods are selected.
- Criterion directions, weights and constant-criterion flags.
- Normalised decision matrix.
- Spearman method-agreement matrix.
- One-at-a-time weight sensitivity.
- First-rank/top-three probabilities and expected rank.
- Complete Excel workbook.
- Colour and black-and-white PNG 600 dpi, SVG and PDF publication figures.
- CSV evidence tables and bundle README.

## Renewable-energy criteria

Potential criteria include capacity factor, IRR, payback, ROI, benefit-cost ratio, direct/indirect employment, regional-development contribution, avoided CO2/pollutants, land use, grid capacity, licensing delay and risk. Definitions, units, reference years and technology-specific comparability must be validated when the 2,000-project dataset is received.
