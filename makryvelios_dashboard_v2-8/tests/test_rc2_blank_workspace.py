from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mapping import fetch_geojson,detailed_offline_map_figure
from ita import CALL_BUDGETS,BENEFICIARY_CATEGORY_CAPS
from gams_compat import preset_definition

def test_no_historical_inputs():
    for name in ("research_questions.csv","research_hypotheses.csv","source_evidence_catalogue.csv","data/rd_projects_reference.xlsx","reference_gams","reference_stata"):
        assert not (ROOT/name).exists(),name
    assert CALL_BUDGETS=={} and BENEFICIARY_CATEGORY_CAPS=={}
    assert preset_definition()["region_caps"]=={}
    source=(ROOT/"app.py").read_text()
    assert "Research programme already encoded" not in source
    assert "frames = {}" in source

def test_neutral_map_without_observations():
    n2=fetch_geojson("NUTS 2 – Regions (13)")
    n3=fetch_geojson("NUTS 3 – Regional units")
    blank=pd.DataFrame({"nuts_id":pd.Series(dtype=str),"value":pd.Series(dtype=float)})
    fig=detailed_offline_map_figure(blank,"value",nuts2_geojson=n2,nuts3_geojson=n3)
    fills=[t for t in fig.data if t.fill=="toself"]
    assert fills and all(t.fillcolor=="#D1D5DB" and "No mapped value" in t.hovertemplate for t in fills)
    assert len(fig.data)>len(fills)
