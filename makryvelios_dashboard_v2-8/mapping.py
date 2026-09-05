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
import plotly.graph_objects as go
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
        mapbox_style="open-street-map", center={"lat": 38.6, "lon": 23.5}, zoom=5.1,
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


# Explicit GAMS-to-Greek-NUTS geography used by the visible ITA/GAMS Studio.
# EP2 in the SYN2 source is "EPANEK2" (a programme budget dimension), not a region,
# and is therefore deliberately left unmapped rather than painted onto Greece.
GAMS_REGION_TO_NUTS2 = {
    "ATT": "EL30",  # Attica
    "NAG": "EL41",  # North Aegean
    "SAG": "EL42",  # South Aegean
    "CRE": "EL43",  # Crete
    "EMK": "EL51",  # Eastern Macedonia and Thrace
    "CMK": "EL52",  # Central Macedonia
    "WMK": "EL53",  # Western Macedonia
    "EPI": "EL54",  # Epirus
    "THE": "EL61",  # Thessaly
    "ION": "EL62",  # Ionian Islands
    "WGR": "EL63",  # Western Greece
    "STE": "EL64",  # Central Greece / Sterea
    "PEL": "EL65",  # Peloponnese
}
NON_GEOGRAPHIC_GAMS_DIMENSIONS = {
    "EP2": "EPANEK2 programme budget dimension (not a geographic NUTS region)",
}


def gams_region_crosswalk(regions: Sequence[str]) -> pd.DataFrame:
    """Return an explicit, auditable GAMS-region to NUTS-2 crosswalk."""
    region_meta = REGIONS.set_index("nuts_id").to_dict("index")
    rows = []
    for region in regions:
        code = str(region).strip().upper()
        nuts = GAMS_REGION_TO_NUTS2.get(code)
        meta = region_meta.get(nuts, {}) if nuts else {}
        rows.append({
            "gams_region": code,
            "nuts_id": nuts,
            "region_el": meta.get("region_el"),
            "region_en": meta.get("region_en"),
            "map_status": "MAPPED" if nuts else "NON-GEOGRAPHIC / UNMAPPED",
            "note": NON_GEOGRAPHIC_GAMS_DIMENSIONS.get(code, "" if nuts else "No verified NUTS-2 crosswalk supplied."),
        })
    return pd.DataFrame(rows)


def _metric_colour(value: float, vmin: float, vmax: float, monochrome: bool = False) -> str:
    if not np.isfinite(value):
        return "#D1D5DB"
    if vmax <= vmin:
        pos = 0.65
    else:
        pos = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    cmap = plt.get_cmap("Greys" if monochrome else "Blues")
    rgba = cmap(0.15 + 0.80 * pos)
    return "#%02X%02X%02X" % tuple(int(round(255 * c)) for c in rgba[:3])


def detailed_offline_map_figure(
    data: pd.DataFrame,
    metric: str,
    *,
    nuts2_geojson: dict,
    nuts3_geojson: dict | None = None,
    title: str | None = None,
    monochrome: bool = False,
    categorical: bool = False,
    category_colours: Mapping[str, str] | None = None,
    show_nuts3: bool = True,
):
    """API-key-free interactive Greece map rendered only from bundled GeoJSON polygons.

    No Mapbox/OSM/Google tile service is contacted.  NUTS-3 boundaries can be overlaid
    as fine internal linework while values are filled at NUTS-2 level.
    """
    if "nuts_id" not in data.columns:
        raise ValueError("Map data must contain nuts_id.")
    values = data.set_index("nuts_id")
    fig = go.Figure()
    category_colours = category_colours or {
        "GREEN": "#16A34A", "GRAY": "#6B7280", "GREY": "#6B7280",
        "RED": "#DC2626", "UNCLASSIFIED": "#64748B",
    }
    numeric_values = pd.to_numeric(data[metric], errors="coerce") if not categorical else pd.Series(dtype=float)
    finite = numeric_values[np.isfinite(numeric_values)].to_numpy(float) if not categorical else np.array([])
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    observed_categories = []
    for feature in nuts2_geojson.get("features", []):
        props = feature.get("properties", {})
        nuts = str(props.get("NUTS_ID") or props.get("id") or "")
        if not nuts.startswith("EL"):
            continue
        if nuts in values.index:
            row = values.loc[nuts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            value = row.get(metric, np.nan)
            region_name = row.get("region_el", row.get("region_en", nuts))
        else:
            value = np.nan if not categorical else "UNCLASSIFIED"
            region_name = props.get("NAME_LATN") or nuts
        if categorical:
            category = str(value).upper() if pd.notna(value) else "UNCLASSIFIED"
            colour = category_colours.get(category, "#CBD5E1")
            observed_categories.append(category)
            value_text = category
        else:
            val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            colour = _metric_colour(float(val) if pd.notna(val) else np.nan, vmin, vmax, monochrome)
            value_text = "No mapped value" if pd.isna(val) else f"{float(val):,.4g}"
        for poly in _geometry_polygons(feature.get("geometry", {})):
            if len(poly) < 3:
                continue
            fig.add_trace(go.Scatter(
                x=poly[:, 0], y=poly[:, 1], mode="lines", fill="toself",
                fillcolor=colour, line=dict(color="#111827", width=1.15),
                name=str(region_name), showlegend=False,
                hovertemplate=f"<b>{region_name}</b><br>NUTS-2: {nuts}<br>{metric}: {value_text}<extra></extra>",
            ))

    if show_nuts3 and nuts3_geojson:
        for feature in nuts3_geojson.get("features", []):
            props = feature.get("properties", {})
            nuts = str(props.get("NUTS_ID") or props.get("id") or "")
            if not nuts.startswith("EL"):
                continue
            for poly in _geometry_polygons(feature.get("geometry", {})):
                if len(poly) < 3:
                    continue
                fig.add_trace(go.Scatter(
                    x=poly[:, 0], y=poly[:, 1], mode="lines",
                    line=dict(color="rgba(15,23,42,0.42)", width=0.55),
                    hoverinfo="skip", showlegend=False,
                ))

    if categorical:
        for cat in ["GREEN", "GRAY", "RED", "UNCLASSIFIED"]:
            if cat in set(observed_categories):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=11, color=category_colours.get(cat, "#CBD5E1")),
                    name=cat, showlegend=True, hoverinfo="skip",
                ))
    elif finite.size:
        # Invisible marker trace provides an accurate continuous legend without external map tiles.
        fig.add_trace(go.Scatter(
            x=[18.55, 18.55], y=[34.55, 34.55], mode="markers",
            marker=dict(
                size=0.1, opacity=0.01, color=[vmin, vmax], cmin=vmin, cmax=vmax,
                colorscale="Greys" if monochrome else "Blues", showscale=True,
                colorbar=dict(title=metric, thickness=14, len=0.65),
            ),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=title or f"Greece · {metric}",
        height=800,
        margin=dict(l=10, r=10, t=58, b=10),
        paper_bgcolor="#0B1F33", plot_bgcolor="#0B1F33",
        font=dict(color="#F8FAFC", size=13),
        legend=dict(bgcolor="rgba(8,21,33,.86)", font=dict(color="#F8FAFC")),
    )
    fig.update_xaxes(range=[18.3, 30.4], visible=False, constrain="domain")
    fig.update_yaxes(range=[34.3, 42.25], visible=False, scaleanchor="x", scaleratio=1)
    return fig


def detailed_static_map_bytes(
    data: pd.DataFrame,
    metric: str,
    *,
    nuts2_geojson: dict,
    nuts3_geojson: dict | None = None,
    title: str | None = None,
    monochrome: bool = False,
    categorical: bool = False,
    category_colours: Mapping[str, str] | None = None,
    show_nuts3: bool = True,
    fmt: str = "png",
    dpi: int = 600,
) -> bytes:
    """Publication export using only bundled vector boundaries.

    SVG/PDF remain vector at arbitrary zoom; PNG defaults to 600 dpi.  NUTS-3
    linework is overlaid to retain coastline/island and internal-border detail.
    """
    if "nuts_id" not in data.columns:
        raise ValueError("Map data must contain nuts_id.")
    values = data.set_index("nuts_id")
    category_colours = category_colours or {
        "GREEN": "#16A34A", "GRAY": "#6B7280", "GREY": "#6B7280",
        "RED": "#DC2626", "UNCLASSIFIED": "#64748B",
    }
    numeric_values = pd.to_numeric(data[metric], errors="coerce") if not categorical else pd.Series(dtype=float)
    finite = numeric_values[np.isfinite(numeric_values)].to_numpy(float) if not categorical else np.array([])
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Greys" if monochrome else "Blues")

    fig, ax = plt.subplots(figsize=(9.2, 9.2), constrained_layout=True)
    for feature in nuts2_geojson.get("features", []):
        props = feature.get("properties", {})
        nuts = str(props.get("NUTS_ID") or props.get("id") or "")
        if not nuts.startswith("EL"):
            continue
        if nuts in values.index:
            row = values.loc[nuts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            value = row.get(metric, np.nan)
        else:
            value = np.nan if not categorical else "UNCLASSIFIED"
        if categorical:
            face = category_colours.get(str(value).upper(), "#D1D5DB")
        else:
            val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            face = cmap(norm(float(val))) if pd.notna(val) else (.86, .88, .90, 1)
        for poly in _geometry_polygons(feature.get("geometry", {})):
            if len(poly) >= 3:
                ax.add_patch(Polygon(poly, closed=True, facecolor=face, edgecolor="#111827", linewidth=0.72, zorder=2))

    if show_nuts3 and nuts3_geojson:
        for feature in nuts3_geojson.get("features", []):
            props = feature.get("properties", {})
            nuts = str(props.get("NUTS_ID") or props.get("id") or "")
            if not nuts.startswith("EL"):
                continue
            for poly in _geometry_polygons(feature.get("geometry", {})):
                if len(poly) >= 3:
                    ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor="#334155", linewidth=0.22, alpha=0.72, zorder=3))

    ax.set_xlim(18.3, 30.4); ax.set_ylim(34.3, 42.25); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"Greece · {metric}", loc="left", fontsize=16, fontweight="bold", pad=12)
    if categorical:
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=category_colours[k], label=k) for k in ["GREEN", "GRAY", "RED", "UNCLASSIFIED"]]
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    elif finite.size:
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap); scalar.set_array([])
        cbar = fig.colorbar(scalar, ax=ax, orientation="horizontal", fraction=.035, pad=.018)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(metric, fontsize=9)
    ax.text(18.38, 34.36, "Offline boundaries: bundled Eurostat GISCO NUTS 2024 · NUTS-3 detail overlay · no API key / map tiles", fontsize=7.2, color="#334155")
    out = io.BytesIO()
    fig.savefig(out, format=fmt, dpi=dpi if fmt.lower() == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out.getvalue()
