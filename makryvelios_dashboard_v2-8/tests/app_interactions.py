from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
at = AppTest.from_file(APP, default_timeout=120)
at.run()
assert not at.exception, at.exception

# Exercise a real OLS estimation using the bundled R&D workbook.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("6. OLS & econometric laboratory")
at.run()
assert not at.exception, at.exception
estimate = next(w for w in at.button if w.label == "Estimate model")
estimate.click()
at.run(timeout=120)
assert not at.exception, at.exception
assert any("Coefficient table" in h.value for h in at.subheader)

# Exercise the new Monte Carlo OLS path with a bounded replication count.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("6A. Monte Carlo & uncertainty")
at.run()
replications = next(w for w in at.slider if w.label == "Replications")
replications.set_value(300)
at.run()
assert not at.exception, at.exception
run_mc = next(w for w in at.button if w.label == "Run Monte Carlo OLS")
run_mc.click()
at.run(timeout=120)
assert not at.exception, at.exception
assert any("Coefficient uncertainty summary" in h.value for h in at.subheader)

# Exercise one-dimensional absorption clustering with automatic k selection.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("10A. Advanced clustering & segmentation")
at.run()
assert not at.exception, at.exception
run_cluster = next(w for w in at.button if w.label == "Run advanced clustering")
run_cluster.click()
at.run(timeout=120)
assert not at.exception, at.exception
assert any("Substantive cluster profiles" in h.value for h in at.subheader)

# Exercise cross-validated predictive modelling on a bounded predictor set.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("10B. Predictive model laboratory")
at.run()
outcome = next(w for w in at.selectbox if w.label == "Continuous outcome")
if "absorption_rate_budget" in outcome.options:
    outcome.set_value("absorption_rate_budget")
at.run()
predictors = next(w for w in at.multiselect if w.label == "Predictors (maximum 200)")
preferred = [c for c in ["Final_Project_s_Budjet_at_the_end_of_the_project", "Project_Duration_year", "Project_start_year"] if c in predictors.options]
if preferred:
    predictors.set_value(preferred)
folds = next(w for w in at.slider if w.label == "Cross-validation folds")
folds.set_value(3)
at.run()
run_predictive = next(w for w in at.button if w.label == "Run predictive comparison")
run_predictive.click()
at.run(timeout=180)
assert not at.exception, at.exception
assert any("Cross-validated model performance" in h.value for h in at.subheader)

# Exercise the panel model UI with region-year aggregation.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("8A. Panel model laboratory")
at.run()
entity = next(w for w in at.selectbox if w.label == "Entity identifier")
entity.set_value("Region")
at.run()
time = next(w for w in at.selectbox if w.label == "Time identifier")
time.set_value("Project_start_year")
outcome = next(w for w in at.selectbox if w.label == "Panel outcome")
outcome.set_value("absorption_rate_budget")
at.run()
panel_predictors = next(w for w in at.multiselect if w.label == "Time-varying predictors")
preferred = [c for c in ["Project_Duration_year", "Final_Project_s_Budjet_at_the_end_of_the_project"] if c in panel_predictors.options]
if preferred:
    panel_predictors.set_value(preferred)
at.run()
run_panel = next(w for w in at.button if w.label == "Estimate panel model suite")
run_panel.click()
at.run(timeout=180)
assert not at.exception, at.exception
assert any("Panel model fit" in h.value for h in at.subheader)

# Exercise the dedicated MCDA module, including Monte Carlo rank acceptability.
module = next(w for w in at.radio if w.label == "Module")
module.set_value("12A. Dedicated MCDA engine")
at.run()
assert not at.exception, at.exception
mcda_draws = next(w for w in at.slider if w.label == "Monte Carlo weight draws")
mcda_draws.set_value(100)
at.run()
run_mcda = next(w for w in at.button if w.label == "Run dedicated MCDA")
run_mcda.click()
at.run(timeout=180)
assert not at.exception, at.exception
assert any("MCDA ranking table" in h.value for h in at.subheader)

print("OLS, Monte Carlo, clustering, predictive, panel and MCDA interactions completed without exceptions")
