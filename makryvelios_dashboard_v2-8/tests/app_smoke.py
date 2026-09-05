from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=90)
source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
assert "POSTDOCTORAL ANALYTICAL ENGINE v5.8.0" in source
assert "12B. Research Command Chair" in source
assert "12A.1 ITA / public-funding decision support" in source
assert "12A.1B GAMS-compatible ITA Studio" in source
assert "12A.2 Expert respondent analytics" in source
assert "12C. Frontier methods laboratory" in source
assert "12D. Agentic Research Mode" in source
assert "ALL v5.7.2 + EARLIER CAPABILITIES RETAINED" in source
assert "Version 5.8.0 documentation library" in source
assert "LLM CO-PILOT · USER API KEY" in source
assert "render_gams_studio" in source
assert "stFileUploaderDropzone" in source
assert "stWidgetLabel" in source
assert "#D8C7FF" in source
assert '[data-baseweb="tab"] *' in source
assert '[data-testid="stAlert"] *' in source
assert "#000000" in source
assert "Example questions and the standard of answer to expect" in source
assert "execute_natural_language_command" in source
assert "Copy-ready questions and analytical commands" in source
assert "How to read and use:" in source
at.run()
assert not at.exception, at.exception
module = next(widget for widget in at.radio if widget.label == "Module")
failures = []
for page in module.options:
    module.set_value(page)
    at.run()
    if at.exception:
        failures.append((page, [str(exc.value) for exc in at.exception]))
    module = next(widget for widget in at.radio if widget.label == "Module")
if failures:
    raise AssertionError(failures)
print(f"{len(module.options)} Streamlit modules rendered without exceptions")
