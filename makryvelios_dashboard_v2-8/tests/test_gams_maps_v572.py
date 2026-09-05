import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mapping import fetch_geojson, gams_region_crosswalk, detailed_static_map_bytes


def test_gams_crosswalk_preserves_ep2_as_non_geographic():
    x = gams_region_crosswalk(["EP2", "ATT", "CMK", "WMK", "STE"])
    assert x.loc[x.gams_region.eq("EP2"), "nuts_id"].isna().all()
    assert x.loc[x.gams_region.eq("ATT"), "nuts_id"].iloc[0] == "EL30"
    assert x.loc[x.gams_region.eq("CMK"), "nuts_id"].iloc[0] == "EL52"


def test_offline_detailed_map_export_bytes():
    n2 = fetch_geojson("NUTS 2 – Regions (13)")
    n3 = fetch_geojson("NUTS 3 – Regional units")
    data = pd.DataFrame({"nuts_id": ["EL30", "EL52", "EL53", "EL64"], "allocated_budget": [4.0, 2.0, 1.0, 3.0]})
    payload = detailed_static_map_bytes(data, "allocated_budget", nuts2_geojson=n2, nuts3_geojson=n3, fmt="png", dpi=120)
    assert payload.startswith(b"\x89PNG")
    assert len(payload) > 5000
