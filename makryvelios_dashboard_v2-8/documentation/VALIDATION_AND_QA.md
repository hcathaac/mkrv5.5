# Validation and Quality-Assurance Protocol

## Automated coverage

`tests/test_core.py` covers data ingestion, descriptive calculations, OLS, high-dimensional output, spatial matching, panel construction, IV, DiD, regularisation, reliability, Monte Carlo, portfolio selection, clustering, prediction, panel estimators, publication bundles, MCDA and AHP.

`tests/app_smoke.py` renders each Streamlit module. `tests/app_interactions.py` exercises retained OLS, Monte Carlo, clustering, predictive and panel paths together with the new dedicated MCDA module.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_core.py
python tests/app_smoke.py
python tests/app_interactions.py
```

## Manual release checks

Test representative CSV/XLSX files, all-sheet ingestion, append and join modes, a detailed regression, a high-dimensional screen, a map, a clustering run, a predictive comparison, an MCDA run and every principal download. Open exported files rather than checking only their byte size.

## Scientific QA

Automated tests do not validate the substantive research design. Every retained result requires checks of analytical unit, missingness, measurement, time order, independence, model assumptions, multiple testing, robustness and causal identification. MCDA additionally requires defensible criteria, directions, value functions, weights and sensitivity analysis.

