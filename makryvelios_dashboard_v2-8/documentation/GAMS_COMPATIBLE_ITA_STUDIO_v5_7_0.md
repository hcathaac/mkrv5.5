# GAMS-Compatible ITA Studio — v5.7.0

## Purpose

The GAMS-Compatible ITA Studio is an additive module. It does not replace Module 12A.1, the respondent-level module, the Research Command Chair or any earlier analytical workflow.

It preserves the algebraic modelling logic used in the supplied Evangelos Makryvelios GAMS files while allowing routine execution with SciPy/HiGHS, avoiding a mandatory commercial GAMS licence.

## Canonical model concepts retained

- project set `p`
- region set `rg`
- sector set `sec`
- intervention set `intv`
- criterion set `crit`
- budget table `budget(p,rg)`
- score table `score(p,crit)`
- binary decision variable `X(p)`
- portfolio objective `PORTFSCORE`
- regional, regional-group, sector and intervention budget constraints
- hard project fixing using the equivalent of `X.fx(GREEN)=1` and `X.fx(RED)=0`
- effective-budget adjustments for ambiguous/newly-green project sets
- GAMS-style model text and `.prn` export
- Monte Carlo score perturbation with reproducible seed

## Presets

### Vangelis – SYN2 540

The preset retains the five regional budget dimensions EP2, ATT, CMK, WMK and STE, ten sectors, three criteria and the supplied Round 1–Round 4 / No ITA weighting matrices. The source discrepancy in the CMK ceiling is flagged rather than silently corrected.

### Vangelis – R&D 2437

The preset retains thirteen regions, less-developed and transition region groups, eight sector ceilings, three intervention ceilings, intervention-specific weights, seed 5780 and the round-specific Monte Carlo perturbation/budget-adjustment logic visible in the supplied source files.

## Execution

The live application labels the execution engine truthfully as SciPy/HiGHS. It does not claim that GAMS has run when it has not. A generated `.gms` model and accompanying `.prn` data files are available for independent GAMS replication.

## Outputs

The Studio provides solver status, portfolio score, selected projects, allocation by region/sector/intervention, binding-constraint diagnostics, project-level decisions, GREEN/GRAY/RED Monte Carlo classifications, coloured Excel output and a complete reproducibility ZIP.

## Original-source visibility

The supplied `.gms` files are bundled under `reference_gams/vangelis/` and can be inspected and downloaded from the Studio. This preserves the researcher's original implementation as a visible methodological reference.


## v5.7.1 mapping and exact-replication update

The SYN2 preset now includes the supplied `budget_syn2.prn`, `score_syn2.prn`, `sector_syn2.prn`, `green1_syn2.txt`, `red1_syn2.txt` and `gray1.txt` as an embedded exact-replication input source. The regional mapper never silently assigns one source column to multiple GAMS regions; missing or duplicate mappings are shown explicitly and solving is blocked until they are resolved.


## v5.7.2 — Offline high-detail Greece maps and GAMS-style diagnostics

The Studio now contains a **Maps & spatial** tab. Maps are rendered directly from the GeoJSON files bundled in `data/`; no Mapbox, Google Maps, OpenStreetMap tile API or API key is required. NUTS-2 regions carry the analytical fill, while bundled NUTS-3 polygons can be overlaid as fine linework for a much more detailed coastline, island and regional-unit outline.

Available mapped outputs include allocated budget, budget utilisation, selected-project exposure, portfolio-score contribution, GREEN/GRAY/RED shares, dominant ITA class and—after Monte Carlo—mean project selection frequency. ITA categorical output uses GREEN `#16A34A`, GRAY `#6B7280` and RED `#DC2626`. Each selected map can be exported as 600-dpi PNG, vector SVG, vector PDF and self-contained interactive HTML.

The geography is explicit and auditable. In SYN2, `ATT`, `CMK`, `WMK` and `STE` map to verified Greek NUTS-2 regions. `EP2` is labelled in the supplied GAMS source as the EPANEK2 programme budget dimension and is intentionally retained outside the map rather than being assigned to a region without evidence. The 2,437-project GAMS region abbreviations are crosswalked to the 13 Greek NUTS-2 regions.

A separate **GAMS diagnostics** tab exposes model status, MIP gap/dual bound/node count when returned by HiGHS, solve time, model cardinalities, active equations, matrix non-zeros, fixed variables, full inequality equation levels/slacks/binding flags, an `X.l`-style variable listing, raw solver messages and parameter snapshots. The reproducibility package additionally contains `.lst`-style solution text and CSV equation/variable listings. Integer-program shadow prices are not fabricated; constraint slack/binding diagnostics are reported instead.
