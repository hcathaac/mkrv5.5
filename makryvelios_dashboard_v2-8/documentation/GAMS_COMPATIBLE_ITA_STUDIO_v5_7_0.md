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
