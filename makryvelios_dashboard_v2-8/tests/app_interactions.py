from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
# Explicit synthetic test fixture; no production bundled dataset or bypass.
# AppTest does not consistently expose file_uploader across Streamlit versions.
import tempfile
source = APP.read_text(encoding="utf-8")
fixture = """    else:
        rng = np.random.default_rng(42)
        n = 240
        years = np.tile(np.arange(2010, 2020), 24)
        budget = rng.uniform(10000, 100000, n)
        duration = rng.integers(1, 6, n)
        frames = {"Synthetic QA fixture": pd.DataFrame({"Region": np.repeat([f"R{i}" for i in range(8)], 30), "Project_start_year": years, "Project_Duration_year": duration, "Final_Project_s_Budjet_at_the_end_of_the_project": budget, "absorption_rate_budget": .2 + budget / 150000 + rng.normal(0,.05,n), "score": rng.normal(size=n)})}
    raw_frames = frames.copy()"""
anchor = "    else:\n        frames = {}\n    raw_frames = frames.copy()"
assert source.count(anchor) == 1
# Put the temporary script beside app.py so __file__-relative assets resolve.
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=APP.parent, delete=False, encoding="utf-8") as tmp:
    tmp.write(source.replace(anchor, fixture)); temporary_app = Path(tmp.name)
import atexit
atexit.register(temporary_app.unlink, missing_ok=True)
at = AppTest.from_file(temporary_app, default_timeout=120)
at.run()
assert not at.exception, at.exception

# Exercise a real OLS estimation using an explicit synthetic test fixture.
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
