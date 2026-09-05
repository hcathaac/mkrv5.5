# Source manifest and analytical provenance

## Supplied application

- `archive/original_app.py` is the exact supplied `app(1).py`, retained for comparison and rollback.
- The new `app.py` is the production entry point.

## Supplied R&D dataset

- `data/rd_projects_reference.xlsx` is the supplied workbook `Makryvelios data for R&D Projects and regional development indicators.xlsx`.
- The workbook contains 3,259 project records and 83 source variables after its embedded second-row header is promoted.
- Twelve regions are observed in the project records; the lookup and maps retain all thirteen Greek NUTS-2 regions.

## Research questions and models

- `research_questions.csv` transcribes the nine formal R&D questions EE1–EE9 from `Ερευνητικά ερωτήματα για Stata.docx` and provides English translations.
- `research_hypotheses.csv` transcribes the nineteen hypotheses attached to EE1–EE9.
- `reference_stata/attica_analysis.do` contains project-level Attica models for EE1, EE2, EE3 and EE5.
- `reference_stata/country_analysis.do` contains the corresponding whole-country project-level models.
- `reference_stata/panel_region_year_analysis.do` constructs the region–year panel and estimates EE1, EE4 and EE6–EE9 with Poisson and negative-binomial robustness specifications.
- The supplied analytical chapters and result workbooks were used to cross-check unit of analysis, variables, estimators and interpretation but are not duplicated in this application archive.
- The additional supplied Makryvelios manuscripts on public-expenditure absorption, research employment, innovation outputs, regional patterns and the revised data-analysis basis were reviewed for model form, transformations, robust-inference practice and reporting language. Their methodological links are recorded in `source_evidence_catalogue.csv`; full manuscripts are not duplicated.
- The Monte Carlo portfolio module is contextually grounded in Mavrotas and Makryvelios (2021), *European Journal of Operational Research*, 291(2), 794–806 (DOI: 10.1016/j.ejor.2020.09.051). The app provides a transparent stochastic cost-benefit implementation rather than claiming to reproduce that paper's complete optimisation architecture.

## Renewable-energy and MCDA extension

- Version 5.2.1 adds a schema-agnostic Dedicated MCDA Engine informed by the supplied 2011 dissertation on multi-criteria evaluation of renewable-energy investments and the supplied 2012 dissertation on renewable-energy projects and support mechanisms.
- The historical dissertation evidence includes project/technology/region comparisons and criteria concerning financial return, capacity factor, employment, regional development, avoided emissions, land and grid constraints.
- The production module implements auditable MAVT, TOPSIS and PROMETHEE-II rankings; equal, user-defined, Entropy, CRITIC and AHP pairwise weighting; AHP consistency; weight perturbation; method agreement; and Monte Carlo rank acceptability.
- The implementation is reusable with future renewable-energy or other project datasets. It does not claim numerical replication of the historical dissertations until the original data, exact criteria functions and elicited judgement matrices are supplied.

## Antonis Tritsis programme

- Prior documented scope: 1,454 approved local-government projects through August 2025; €3.76 billion approved budget; all thirteen Greek NUTS-2 regions; Ministry of the Interior source fields covering beneficiary, beneficiary type, region, thematic call, approval date and approved budget.
- `AT-RQ1` to `AT-RQ11` are clearly labelled reconstructions from that documented dataset scope, not verbatim questions from the R&D source document.
- `AT-RQ12` is a new cross-programme research question linking the regional R&D and Antonis Tritsis evidence.
- The Antonis Tritsis project-level workbook was not among the final attached files, so it is not bundled. It can be uploaded directly as XLSX or CSV without code changes.

## Geography

- `data/greece_nuts2_2024.geojson` and `data/greece_nuts3_2024.geojson` are Greek subsets of Eurostat GISCO NUTS 2024 1:1 million boundary files.
- `data/greece_region_lookup.csv` contains bilingual names, NUTS-2 identifiers and centroids for the thirteen Greek regions.

## Reproducibility boundary

The application reproduces the model families and provides auditable exports. Numerical equality with earlier Stata/R tables requires the same dataset version, recodes, missing-data decisions, sample restriction, time allocation, fixed effects and covariance estimator. The app surfaces these decisions rather than concealing them.

## Research Command Chair extension

- Version 5.3.0 adds `research_chair.py` as an isolated free/offline protocol and evidence-scoping layer; v5.3.1 fixes its all-numeric default and command execution; v5.3.2 adds ranked headline-finding interpretation and stricter light-surface text contrast; v5.4.0 adds the full prompt library, bounded advanced command intents and output-by-output interpretation guides; v5.5.0 adds the one-screen batch-question autopilot, feasibility verdicts, genuine paper prose and complete interactive/high-resolution Chair figures.
- Spreadsheet evidence is selected from the existing active dataset without mutating the source frame.
- PDF text is extracted locally by page; the selected document, page interval and keyword scope is exported with the results.
- Computable derived variables use a restricted mathematical syntax rather than arbitrary code execution.
- Optional Ollama requests target only a locally running endpoint. No paid or hosted AI service is required.
- Custom prose instructions are documented but are not represented as validated executable estimators.
- `prompt_library.py` contains the auditable copy-ready prompt catalogue and identifies the validated module in which each request should be executed.
- `output_guidance.py` generates the plain-language explanation layer used by the interface, spreadsheets, reports and publication bundles.

## Documentation set

- `documentation/COMPLETE_DOCUMENTATION.md` is the authoritative consolidated technical and user documentation.
- The DOCX and PDF editions are rendered deliverables of the same controlled source.
- Focused quick-start, MCDA, Research Command Chair, deployment, validation and requirements guides support operational use without replacing the complete documentation.
- Module 13 exposes the documentation files for direct download when they are present in the deployed package.

## v5.6.1 respondent / ITA bridge and v5.7.0 GAMS-compatible extension

- The v5.6.1 application remains the preserved functional base: the existing ITA/public-funding module, expert respondent analytics, empirical-weight bridge, Research Command Chair, mapping, econometrics, MCDA, Monte Carlo and export paths remain present.
- `gams_compat.py` adds a solver-neutral GAMS-style binary portfolio model compiled to SciPy/HiGHS for licence-free execution; it does not replace or delete `ita.py`.
- `gams_ui.py` adds the separate `12A.1B GAMS-compatible ITA Studio` while retaining the existing `12A.1` and `12A.2` modules.
- `reference_gams/vangelis/` contains the supplied original `.gms` sources as an auditable reference library. The application keeps the original GAMS formulation visible and supports generated `.gms`, `.prn`, status lists and complete reproducibility packages.
- The built-in presets preserve the supplied SYN2 540-project rounds and the 2,437-project R&D intervention/Monte-Carlo structure. Source discrepancies are preserved and surfaced rather than silently corrected.
- `llm_bridge.py` provides an optional user-key interpretation layer. API credentials are held only in Streamlit session state and are not written to configuration files or exports. The LLM is not used as the mathematical solver and cannot silently alter portfolio decisions.
- The default numerical execution remains deterministic/open-source (SciPy/HiGHS). Native GAMS remains an external compatibility/replication route if a licence is available later.

## GAMS-compatible geography and diagnostics — v5.7.2

- The GAMS Studio uses the already bundled `data/greece_nuts2_2024.geojson` and `data/greece_nuts3_2024.geojson`; mapping does not require a map API key or external tile service.
- Explicit GAMS/NUTS crosswalks are coded only where the source abbreviation is geographically identifiable. In SYN2, `EP2` is documented in the supplied `.gms` source as the EPANEK2 programme budget and is therefore kept outside the geographic map rather than assigned to a Greek region.
- High-detail publication maps fill NUTS-2 analytical regions and overlay NUTS-3 vector linework for coastline, island and regional-unit detail. Vector SVG/PDF exports retain boundary detail independently of raster DPI.
- GAMS-style solver/equation/variable listings are generated from the deterministic HiGHS solution. Integer-program shadow prices are not fabricated.
