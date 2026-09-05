# Makryvelios Workbench v5.7.0 — local validation record

Release date: 5 September 2026

## Baseline preservation

The uploaded GitHub package was identified as v5.6.1. A file-by-file preservation check against the untouched backup found **0 missing baseline files** after the v5.7.0 upgrade. The upgrade is additive; the pre-existing ITA, respondent, econometrics, mapping, MCDA, Research Command Chair, reporting and export modules remain in place.

## Added implementation

- `gams_compat.py` — GAMS-style binary portfolio model with SciPy/HiGHS execution, constraints, fixed variables, scenario weights, Monte Carlo and reproducibility/GAMS exports.
- `gams_ui.py` — separate 12A.1B GAMS-Compatible ITA Studio.
- `llm_bridge.py` — optional session-only user API-key LLM co-pilot for interpretation and drafting, kept outside numerical optimisation.
- `reference_gams/vangelis/` — ten original supplied GAMS model files retained verbatim as the visible reference library.
- High-contrast UI guard and visible LLM API-key controls in `app.py`.

## Validation performed in the packaging sandbox

- Python compilation: PASS for the application and new modules.
- Baseline file preservation: PASS (0 missing files).
- New GAMS-compatible backend / export test: PASS.
- SYN2 original-round and Monte Carlo reproducibility test: PASS.
- User-key LLM configuration test: PASS.
- Existing core tests before the optional panel dependency: PASS (22 tests).
- Existing remaining core tests after the panel test: PASS (13 tests).

One existing panel-econometrics test could not execute in the packaging sandbox because the sandbox does not have the `linearmodels` package installed. `requirements.txt` already contains `linearmodels>=6,<8`, so no application dependency change is required. The sandbox also lacks the `streamlit` package, therefore Streamlit AppTest smoke execution was not possible locally; `requirements.txt` already contains `streamlit>=1.36,<2`.

## Deployment path

Keep the existing application directory and Streamlit entry point:

`makryvelios_dashboard_v2-8/app.py`

Do not create an extra nested application directory when replacing files in GitHub.
