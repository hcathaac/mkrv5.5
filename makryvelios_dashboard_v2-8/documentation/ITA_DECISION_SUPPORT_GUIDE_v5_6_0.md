# ITA / Public-Funding Decision Support — v5.6.0

## Scope

Module **12A.1** extends the retained Makryvelios v5.5.3 workbench with an auditable decision-support layer for public-funding portfolios. It does not alter uploaded data, replace administrative judgement or claim that model selection alone establishes policy effectiveness.

## Required data mapping

Map a unique project identifier, call/invitation, beneficiary or municipality, requested budget, and at least two ordered criteria. For the Antonis Tritsis application, map the defined criteria in C1-C6 order, with C1 representing developmental need/equity. Optional fields activate eligibility gates, disadvantaged-area rules, beneficiary categories M1-M7, observed allocation, geography adjustments and Greek region profiles.

Every source value remains unchanged. The engine builds a separate model-ready table that records source row, mapped fields, normalised weights, eligibility, scores, ranks and all assumptions.

## ITA-PB

ITA-PB exposes four policy rounds:

1. Pure weighted-score portfolio optimisation.
2. C1 policy priority.
3. C1 priority plus a minimum disadvantaged-area funding share.
4. Full policy optimisation including beneficiary-category caps.

Projects are labelled policy-robust green/red, equity-sensitive gain/loss or policy-conflict. These classes show sensitivity to activated rules; they do not label projects as politically desirable or undesirable.

## Hybrid ITA-RW

Hybrid ITA-RW jointly evaluates score and weight uncertainty. Scores are sampled within the selected uncertainty interval, published converging-weight scenarios narrow toward the centre, and a binary feasible portfolio is solved repeatedly for every round. Projects above the Green threshold are frozen in, projects below the Red threshold are frozen out, and Gray projects continue to the next round. The default thresholds are 95% and 5%; the seed and every scenario setting are exported.

The v5.6.0 baseline uses mapped central weights and the documented converging-weight construction. From v5.6.1, Module 12A.2 can supply a validated respondent-level empirical weight matrix. Empirical mode samples complete respondent vectors and progressively contracts them towards their empirical centre; it does **not** invent missing distributions or the papers' final confirmatory models.

## Visual decision support

Colours have stable meanings throughout the module:

- Green: robust inclusion or selected funding.
- Gray: unresolved project requiring another round or explicit review.
- Red: robust exclusion.
- Blue/purple/amber: score sensitivity, weight sensitivity and combined sensitivity.

The round matrix shows inclusion probability for every project and round; the Sankey-style infographic shows Green/Gray/Red movement; the score-weight plane separates the two uncertainty sources; and funding utilisation shows allocated versus unallocated call envelopes. Full values remain downloadable even when the screen matrix is limited for legibility.

## Geography and spatial analysis

Recognised Greek NUTS-2 names or codes activate the no-key national allocation map. Global and local Moran diagnostics use a three-nearest-neighbour structure to accommodate island geography. They are exploratory and depend on the neighbourhood definition. Funding totals must not be interpreted as need, impact or effectiveness; those interpretations require population, socioeconomic or other defensible denominators.

## Optimisation and replication

The live application solves the exact binary model with SciPy/HiGHS. Call funding envelopes, eligibility, optional beneficiary caps, frozen decisions and the equity floor are enforced as constraints. Each ITA export contains model-ready projects, every round, probabilities, converging weights, allocations, scorecards, settings and an independent GAMS model/data route. A GAMS or GUROBI licence is not required for the live dashboard.

## Safe workflow

1. Audit the uploaded files and confirm the unit of analysis.
2. Verify C1-C6 direction, scale and missing-data treatment.
3. Confirm call envelopes, eligibility fields, caps and equity definition.
4. Run ITA-PB to reveal policy-rule sensitivity.
5. Run Hybrid ITA-RW with a small scenario count for workflow testing.
6. Increase simulations for publication and document convergence.
7. Review the round matrix, Gray zone, score-weight plane and geographic patterns.
8. Export the complete evidence package before drafting conclusions.

## Known source discrepancy

The supplied methodological note states 2,929 projects, while its invitation table totals 2,928. The application never forces either number: it uses the exact uploaded row count, displays a warning when it matches either figure and records N in every export.
