"""Fetch and retain only Greek features from official GISCO files."""
from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
targets = {
    "greece_nuts2_2024.geojson": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2024_4326_LEVL_2.geojson",
    "greece_nuts3_2024.geojson": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2024_4326_LEVL_3.geojson",
}
for filename, url in targets.items():
    source = requests.get(url, timeout=90)
    source.raise_for_status()
    obj = source.json()
    features = [f for f in obj.get("features", []) if f.get("properties", {}).get("CNTR_CODE") == "EL"]
    out = {"type": "FeatureCollection", "name": obj.get("name", filename), "features": features}
    path = ROOT / "data" / filename
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(filename, len(features), path.stat().st_size)
