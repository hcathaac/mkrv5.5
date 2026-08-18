"""Greek geography, spatial diagnostics and publication-quality map exports."""
from __future__ import annotations

import io
import json
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


GISCO_URLS = {
    "NUTS 2 – Regions (13)": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2024_4326_LEVL_2.geojson",
    "NUTS 3 – Regional units": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2024_4326_LEVL_3.geojson",
}

LOCAL_BOUNDARIES = {
    "NUTS 2 – Regions (13)": Path(__file__).resolve().parent / "data" / "greece_nuts2_2024.geojson",
    "NUTS 3 – Regional units": Path(__file__).resolve().parent / "data" / "greece_nuts3_2024.geojson",
}

REGIONS = pd.DataFrame([
    ("EL30", "Αττική", "Attica", 37.98, 23.73),
    ("EL41", "Βόρειο Αιγαίο", "North Aegean", 39.10, 26.55),
    ("EL42", "Νότιο Αιγαίο", "South Aegean", 36.65, 25.20),
    ("EL43", "Κρήτη", "Crete", 35.24, 24.81),
    ("EL51", "Ανατολική Μακεδονία και Θράκη", "Eastern Macedonia and Thrace", 41.13, 25.40),
    ("EL52", "Κεντρική Μακεδονία", "Central Macedonia", 40.64, 22.95),
    ("EL53", "Δυτική Μακεδονία", "Western Macedonia", 40.30, 21.60),
    ("EL54", "Ήπειρος", "Epirus", 39.66, 20.85),
    ("EL61", "Θεσσαλία", "Thessaly", 39.55, 22.20),
    ("EL62", "Ιόνια Νησιά", "Ionian Islands", 38.65, 20.55),
    ("EL63", "Δυτική Ελλάδα", "Western Greece", 38.25, 21.45),
    ("EL64", "Στερεά Ελλάδα", "Central Greece", 38.60, 22.65),
    ("EL65", "Πελοπόννησος", "Peloponnese", 37.50, 22.35),
], columns=["nuts_id", "region_el", "region_en", "lat", "lon"])


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value).casefold())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.replace("-", " ").replace("&", "and").split())


ALIASES = {}
for row in REGIONS.itertuples():
    for alias in (row.nuts_id, row.region_el, row.region_en):
        ALIASES[_fold(alias)] = row.nuts_id
ALIASES.update({
    "attiki": "EL30", "anatoliki makedonia thraki": "EL51", "east macedonia and thrace": "EL51",
    "kentriki makedonia": "EL52", "dytiki makedonia": "EL53", "ipeiros": "EL54",
    "thessalia": "EL61", "ionia nisia": "EL62", "dytiki ellada": "EL63", "sterea ellada": "EL64",
    "peloponnisos": "EL65", "voreio aigaio": "EL41", "notio aigaio": "EL42", "kriti": "EL43",
})


def match_nuts2(series: pd.Series) -> pd.Series:
    return series.map(lambda x: ALIASES.get(_fold(x)) if pd.notna(x) else np.nan)


def fetch_geojson(level: str, custom_bytes: bytes | None = None) -> dict:
    if custom_bytes:
        obj = json.loads(custom_bytes.decode("utf-8-sig"))
    elif LOCAL_BOUNDARIES[level].exists():
        obj = json.loads(LOCAL_BOUNDARIES[level].read_text(encoding="utf-8"))
    else:
        url = GISCO_URLS[level]
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        obj = response.json()
    features = []
    for feature in obj.get("features", []):
        props = feature.get("properties", {})
        code = props.get("CNTR_CODE") or props.get("CNTR_ID")
        nuts = props.get("NUTS_ID") or props.get("id")
        if code == "EL" or str(nuts).startswith("EL"):
            features.append(feature)
    if not features:
        raise ValueError("The boundary file contains no Greek features.")
    return {"type": "FeatureCollection", "features": features}


def aggregate_geography(
    df: pd.DataFrame,
    geography: str,
    metric: str,
    aggregation: str,
    level: str,
) -> pd.DataFrame:
    d = df[[geography, metric]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    if level.startswith("NUTS 2"):
        d["nuts_id"] = match_nuts2(d[geography])
        if d.nuts_id.notna().sum() == 0:
            d["nuts_id"] = d[geography].astype("string")
        key = "nuts_id"
    else:
        d["nuts_id"] = d[geography].astype("string").str.strip().str.upper()
        key = "nuts_id"
    funcs = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count", "Minimum": "min", "Maximum": "max"}
    out = d.groupby(key, dropna=False)[metric].agg(funcs[aggregation]).reset_index()
    if level.startswith("NUTS 2"):
        out = REGIONS.merge(out, on="nuts_id", how="left")
    return out


def choropleth_figure(data: pd.DataFrame, geojson: dict, metric: str, monochrome: bool = False):
    scale = "Greys" if monochrome else [[0, "#f7fbff"], [.25, "#c6dbef"], [.5, "#6baed6"], [.75, "#2171b5"], [1, "#08306b"]]
    fig = px.choropleth_mapbox(
        data, geojson=geojson, locations="nuts_id", featureidkey="properties.NUTS_ID",
        color=metric, hover_name="region_el" if "region_el" in data else "nuts_id",
        hover_data={metric: ":,.3f", "nuts_id": True}, color_continuous_scale=scale,
        mapbox_style="carto-positron", center={"lat": 38.6, "lon": 23.5}, zoom=5.1,
        opacity=.83, height=760,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), title=f"Greece: {metric}", font=dict(family="Arial", size=13))
    return fig


def _geometry_polygons(geometry: dict) -> list[np.ndarray]:
    if geometry.get("type") == "Polygon":
        return [np.asarray(ring) for ring in geometry.get("coordinates", [])[:1]]
    if geometry.get("type") == "MultiPolygon":
        return [np.asarray(poly[0]) for poly in geometry.get("coordinates", []) if poly]
    return []


def static_map_bytes(data: pd.DataFrame, geojson: dict, metric: str, monochrome: bool = False, fmt: str = "png", dpi: int = 600) -> bytes:
    values = data.set_index("nuts_id")[metric].to_dict()
    finite = np.asarray([v for v in values.values() if np.isfinite(v)], dtype=float)
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmin == vmax:
        vmax = vmin + 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Greys" if monochrome else "Blues")
    patches, colours = [], []
    for feature in geojson.get("features", []):
        nuts = feature.get("properties", {}).get("NUTS_ID")
        value = values.get(nuts, np.nan)
        for coordinates in _geometry_polygons(feature.get("geometry", {})):
            if len(coordinates) >= 3:
                patches.append(Polygon(coordinates, closed=True))
                colours.append(cmap(norm(value)) if np.isfinite(value) else (.88, .88, .88, 1))
    fig, ax = plt.subplots(figsize=(8.27, 9.4), constrained_layout=True)
    collection = PatchCollection(patches, facecolor=colours, edgecolor="#222222", linewidth=.35)
    ax.add_collection(collection)
    ax.set_xlim(18.5, 30.2); ax.set_ylim(34.5, 42.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Greece: {metric}", loc="left", fontsize=16, fontweight="bold", pad=12)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, orientation="horizontal", fraction=.035, pad=.02)
    cbar.ax.tick_params(labelsize=9)
    ax.text(18.55, 34.55, "Boundaries: Eurostat GISCO NUTS 2024. Missing areas shown in grey.", fontsize=7.5, color="#444444")
    out = io.BytesIO()
    fig.savefig(out, format=fmt, dpi=dpi if fmt == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out.getvalue()


def knn_weights(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    coords = df[["lat", "lon"]].to_numpy(float)
    n = len(coords)
    if n < 2:
        raise ValueError("At least two mapped areas are required.")
    k = min(max(1, k), n - 1)
    distance = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(distance[i])[:k]] = 1
    return W / W.sum(axis=1, keepdims=True)


def moran_diagnostics(data: pd.DataFrame, metric: str, permutations: int = 999, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = data.dropna(subset=[metric, "lat", "lon"]).copy()
    if len(d) < 4:
        raise ValueError("At least four mapped regions are required.")
    y = d[metric].to_numpy(float)
    W = knn_weights(d, k)
    z = (y - y.mean()) / (y.std(ddof=0) or 1)
    I = float(z @ W @ z / (z @ z))
    rng = np.random.default_rng(42)
    sims = np.asarray([float((p := rng.permutation(z)) @ W @ p / (p @ p)) for _ in range(permutations)])
    p_global = (np.sum(np.abs(sims) >= abs(I)) + 1) / (permutations + 1)
    lag = W @ z
    local_i = z * lag
    local_sims = np.empty((permutations, len(z)))
    for j in range(permutations):
        p = rng.permutation(z)
        local_sims[j] = p * (W @ p)
    local_p = (np.sum(np.abs(local_sims) >= np.abs(local_i), axis=0) + 1) / (permutations + 1)
    cluster = np.select(
        [(z >= 0) & (lag >= 0), (z < 0) & (lag < 0), (z >= 0) & (lag < 0)],
        ["High–High", "Low–Low", "High–Low"], default="Low–High",
    )
    local = d[[c for c in ["nuts_id", "region_el", "region_en", metric] if c in d]].copy()
    local["z_score"] = z; local["spatial_lag_z"] = lag; local["local_moran_i"] = local_i
    local["permutation_p"] = local_p; local["cluster"] = cluster; local["significant_5pct"] = local_p < .05
    global_table = pd.DataFrame([{
        "diagnostic": "Global Moran's I", "value": I, "permutation_p": p_global,
        "permutations": permutations, "weights": f"{k}-nearest-neighbour, row-standardised",
        "interpretation": "Positive values indicate clustering of similar values; negative values indicate spatial dispersion."
    }])
    return global_table, local


def map_commentary(global_table: pd.DataFrame, local: pd.DataFrame, metric: str) -> list[str]:
    row = global_table.iloc[0]
    direction = "positive clustering" if row.value > 0 else "spatial dispersion"
    significance = "statistically detectable" if row.permutation_p < .05 else "not statistically distinguishable from spatial randomness"
    comments = [f"For {metric}, Global Moran's I is {row.value:.3f} (permutation p={row.permutation_p:.3f}), indicating {direction} that is {significance} at the 5% level."]
    sig = local[local.significant_5pct]
    if sig.empty:
        comments.append("No individual region is flagged as a significant local spatial cluster at 5%; apparent map contrasts should therefore be treated descriptively.")
    else:
        for cluster, group in sig.groupby("cluster"):
            names = ", ".join(group.get("region_en", group.nuts_id).astype(str))
            comments.append(f"Significant {cluster} pattern: {names}.")
    comments.append("Spatial diagnostics are exploratory and depend on the neighbourhood definition; Greek islands justify KNN weights, with contiguity/custom weights advisable as a robustness check.")
    return comments
