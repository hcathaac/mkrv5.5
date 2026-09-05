from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "app.py").read_text(encoding="utf-8")


def test_all_previous_modules_retained_and_new_modules_added():
    retained = [
        "1. Executive overview",
        "2. Data hub & audit",
        "3. Research questions",
        "4. Descriptive statistics",
        "5. Hypothesis tests",
        "6. OLS & econometric laboratory",
        "6A. Monte Carlo & uncertainty",
        "7. 1,000 × 1,000 batch engine",
        "8. Original R&D regional panel",
        "8A. Panel model laboratory",
        "9. Detailed Greece GIS",
        "10. Time series & multivariate",
        "10A. Advanced clustering & segmentation",
        "10B. Predictive model laboratory",
        "11. Publication figures & HTML report",
        "12. Scenario & allocation engine",
        "12A. Dedicated MCDA engine",
        "12A.1 ITA / public-funding decision support",
        "12A.1B GAMS-compatible ITA Studio",
        "12A.2 Expert respondent analytics",
        "12B. Research Command Chair",
        "13. Methods & reproducibility",
    ]
    for label in retained:
        assert label in source
    assert "12C. Frontier methods laboratory" in source
    assert "12D. Agentic Research Mode" in source
    assert "ALL v5.7.2 + EARLIER CAPABILITIES RETAINED" in source


def test_previous_core_files_still_exist():
    expected = [
        "analytics_core.py", "advanced_analytics.py", "mcda.py", "ita.py", "ita_ui.py",
        "gams_compat.py", "gams_ui.py", "respondent.py", "respondent_ui.py",
        "mapping.py", "research_chair.py", "llm_bridge.py", "visuals.py", "reporting.py",
    ]
    for name in expected:
        assert (ROOT / name).exists(), name
