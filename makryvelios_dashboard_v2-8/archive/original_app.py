"""
R&D Projects and Regional Development Dashboard
Streamlit + Python econometrics + optional GIS/spatial diagnostics.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
RAW_XLSX = DATA_DIR / "rd_projects_raw.xlsx"
CLEAN_CSV = DATA_DIR / "rd_projects_clean.csv"
CATALOG_CSV = DATA_DIR / "variable_catalog.csv"
MODEL_SPECS_CSV = DATA_DIR / "model_specifications.csv"
REGION_LOOKUP_CSV = DATA_DIR / "greece_region_lookup.csv"

GISCO_NUTS2_GEOJSON = (
    "https://gisco-services.ec.europa.eu/distribution/v1/nuts-2024/geojson/"
    "NUTS_RG_01M_2024_4326_LEVL_2.geojson"
)

st.set_page_config(
    page_title="R&D Regional Econometrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Data loading and utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
    else:
        df = pd.read_excel(RAW_XLSX, sheet_name="Data (all)", header=1)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    return df

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_CSV)

@st.cache_data(show_spinner=False)
def load_model_specs() -> pd.DataFrame:
    return pd.read_csv(MODEL_SPECS_CSV)

@st.cache_data(show_spinner=False)
def load_region_lookup() -> pd.DataFrame:
    return pd.read_csv(REGION_LOOKUP_CSV)


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x:,.0f}"
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    return f"{x:.{digits}f}"


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def available(cols: List[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]


def clean_formula_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def add_log_columns(d: pd.DataFrame, vars_: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    d = d.copy()
    mapping = {}
    for v in vars_:
        if v not in d.columns:
            continue
        if pd.api.types.is_numeric_dtype(d[v]) and d[v].min(skipna=True) >= 0:
            new = f"ln_{clean_formula_name(v)}"
            d[new] = np.log1p(d[v])
            mapping[v] = new
        else:
            mapping[v] = v
    return d, mapping

# -----------------------------
# Variable blocks
# -----------------------------
PROJECT_CONTROLS_CAT = [
    "rtdi_sector", "rtdi_subsector", "scientific_field", "type_of_beneficiary",
    "type_of_research_organization", "role_of_beneficiary", "region",
]
PROJECT_CONTROLS_NUM = [
    "project_duration_year", "final_projects_budget_at_the_end_of_the_project",
    "final_public_expenditure_at_the_end", "pct_absorption_rate_per_budget",
]

ECONOMIC_BLOCK = [
    "gdp_region_start_year", "gdp_region_end_year",
    "gdp_region_per_person_start_year", "gdp_region_per_person_end_year",
    "gross_value_added_region_start_year", "gross_value_added_region_end_year",
    "employment_region_start_year", "employment_region_end_year",
    "number_of_business_region_start_year", "number_of_business_region_end_year",
]
RD_HUMAN_BLOCK = [
    "employment_r_d_number_region_start_year", "employment_r_d_number_region_end_year",
    "employment_r_d_ipa_region_start_year", "employment_r_d_ipa_region_end_year",
    "researchers_number_region_start_year", "researchers_numbers_region_end_year",
    "researchers_ipa_region_start_year", "researchers_ipa_region_end_year",
    "educational_institution_per_region", "research_center_per_region",
]
RD_EXP_BLOCK = [
    "r_d_exp_region_start_year", "r_d_exp_region_end_year", "r_d_per_inhabitant_region_2015",
    "r_d_exp_int_region_start_year", "r_d_exp_int_region_end_year",
    "intensity_r_d_exp_start_year", "intensity_r_d_exp_end_year",
]
INNOVATION_BLOCK = [
    "percent_innov_busi_region_2010_2012", "percent_company_innov_prod_region_2010_2012",
    "exp_innov_act_region_2012", "rate_busin_region_newproducts",
    "rate_turnover_innovative_products_region_2012",
    "rate_busin_region_collaborations_with_any_organiz",
    "rate_busin_region_innovating_or_marketing_2012_2014",
]
SCIENCE_BLOCK = [
    "numb_sc_pub_per_million_rd_region_2011", "numb_sc_pub_per_ipa_region_2011",
    "impact_index_of_publications_by_region_2010_2014",
    "numb_sc_pub_per_million_rd_region_2010_2014", "numb_sc_pub_per_ipa_region_2015",
    "numb_sc_pub_per_region_2010_2014", "numb_report_sc_pub_region_2010_2014",
    "numb_inter_collab_per_region_2010_2014", "numb_sc_pub_per_region_2012_2016",
    "numb_sc_pub_per_region_2012_2016_1", "numb_sc_pub_per_region_2012_2016_2",
    "numb_sc_pub_per_region_2014_2016",
]
PROJECT_OUTPUTS = [
    "indicator_1_new_jobs_ipa", "indicator_3_new_jobs_womens_ipa",
    "indicator_4_number_of_rdti_projects", "indicator_5_num_coop_company_research_instit",
    "indicator_6_num_of_research_job_ipa", "indicator_9_num_of_jobs_ipa_smes",
    "indicator_10_induced_investments", "indicator_501_jobs_dur_the_operat_ipa",
    "indicator_3103_num_company_res_lab_busin_collab", "indicator_3104_num_new_business_rtd",
    "indicator_3106_num_company_benef", "indicator_3107_num_smes_com_benef",
    "indicator_3110_num_of_smes_benef", "indicator_3111_num_spin_off_spin_outs",
    "indicator_3112_num_of_joint_project_other_countries", "indicator_3115_num_of_patent",
    "indicator_3121_num_of_lab_supp", "indicator_3204_num_of_business_will_be_supported",
    "indicator_6913_num_res_part", "indicator_8004_stud_research_evaluat", "indicator_8009_num_events",
]

BLOCKS = {
    "Economic development / business base": ECONOMIC_BLOCK,
    "R&D human capital / institutions": RD_HUMAN_BLOCK,
    "R&D expenditure / intensity": RD_EXP_BLOCK,
    "Business innovation / cooperation": INNOVATION_BLOCK,
    "Scientific output / networks": SCIENCE_BLOCK,
    "Project financial controls": PROJECT_CONTROLS_NUM,
}

# -----------------------------
# Panel construction
# -----------------------------
def build_region_year_panel(df: pd.DataFrame, mode: str = "Start-year allocation") -> pd.DataFrame:
    d = df.copy()
    budget = "final_projects_budget_at_the_end_of_the_project"
    public = "final_public_expenditure_at_the_end"
    start = "project_start_year"
    end = "project_end_year"

    if mode.startswith("Active"):
        rows = []
        for _, r in d.iterrows():
            if pd.isna(r[start]) or pd.isna(r[end]):
                continue
            years = list(range(int(r[start]), int(r[end]) + 1))
            n = max(len(years), 1)
            for y in years:
                rr = r.copy()
                rr["year"] = y
                rr["allocation_weight"] = 1.0 / n
                rows.append(rr)
        d = pd.DataFrame(rows)
    elif mode.startswith("End"):
        d["year"] = d[end]
        d["allocation_weight"] = 1.0
    else:
        d["year"] = d[start]
        d["allocation_weight"] = 1.0

    # allocated outputs for flow/sum variables
    sum_cols = available(PROJECT_OUTPUTS + [budget, public], d)
    for c in sum_cols:
        d[f"{c}_alloc"] = d[c].fillna(0) * d["allocation_weight"]

    keys = ["region", "nuts_id", "region_el", "lat", "lon", "year"]
    agg = {
        "a_a_project": "nunique",
        "pct_absorption_rate_per_budget": "mean",
        "project_duration_year": "mean",
    }
    for c in available(ECONOMIC_BLOCK + RD_HUMAN_BLOCK + RD_EXP_BLOCK + INNOVATION_BLOCK + SCIENCE_BLOCK, d):
        agg[c] = "mean"
    for c in sum_cols:
        agg[f"{c}_alloc"] = "sum"

    panel = d.groupby(keys, dropna=False).agg(agg).reset_index()
    panel = panel.rename(columns={
        "a_a_project": "project_count",
        "pct_absorption_rate_per_budget": "absorption_rate_budget_mean",
        "project_duration_year": "project_duration_mean",
        f"{budget}_alloc": "total_budget_allocated",
        f"{public}_alloc": "total_public_expenditure_allocated",
        "indicator_5_num_coop_company_research_instit_alloc": "collaborative_projects",
        "indicator_3115_num_of_patent_alloc": "patents",
        "indicator_3111_num_spin_off_spin_outs_alloc": "spin_offs",
        "indicator_3110_num_of_smes_benef_alloc": "smes_benefited",
        "indicator_3106_num_company_benef_alloc": "participating_firms",
    })
    panel["mean_budget"] = panel["total_budget_allocated"] / panel["project_count"].replace(0, np.nan)
    panel["mean_public_expenditure"] = panel["total_public_expenditure_allocated"] / panel["project_count"].replace(0, np.nan)
    panel["public_share_of_budget"] = panel["total_public_expenditure_allocated"] / panel["total_budget_allocated"].replace(0, np.nan)
    return panel

# -----------------------------
# Econometrics
# -----------------------------
def make_formula(y: str, x_vars: List[str], cats: List[str] | None = None, fe_region=False, fe_year=False) -> str:
    rhs = []
    rhs.extend(x_vars)
    if cats:
        rhs.extend([f"C({c})" for c in cats])
    if fe_region:
        rhs.append("C(region)")
    if fe_year:
        rhs.append("C(year)")
    rhs = " + ".join(dict.fromkeys(rhs)) if rhs else "1"
    return f"{y} ~ {rhs}"


def fit_model(data: pd.DataFrame, y: str, x: List[str], model_type: str, transform_y: str,
              transform_x: bool, cats: List[str], fe_region: bool, fe_year: bool,
              cluster: str | None) -> Tuple[object, pd.DataFrame, str]:
    # De-duplicate required columns. Duplicate column selection in pandas returns
    # repeated DataFrame columns; Patsy/statsmodels then sees categorical variables
    # as 2D objects and fails. This is common when region is both a categorical
    # control and the cluster variable.
    needed = list(dict.fromkeys([y] + x + cats + ([cluster] if cluster else [])))
    if fe_region and "region" not in needed and "region" in data.columns:
        needed.append("region")
    if fe_year and "year" not in needed and "year" in data.columns:
        needed.append("year")
    d = data[available(needed, data)].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        raise ValueError("No complete observations remain after dropping missing values.")

    if transform_y == "log1p":
        if d[y].min(skipna=True) < 0:
            raise ValueError("log1p cannot be applied to a dependent variable with negative values.")
        d["__y__"] = np.log1p(d[y])
        y_formula = "__y__"
    elif transform_y == "fractional_clip":
        d["__y__"] = d[y].clip(0, 1)
        y_formula = "__y__"
    else:
        y_formula = y

    if transform_x:
        d, mp = add_log_columns(d, x)
        x_formula = [mp.get(v, v) for v in x]
    else:
        x_formula = x

    formula = make_formula(y_formula, x_formula, cats, fe_region, fe_year)
    cov_kw = {}
    cov_type = "nonrobust"
    if cluster and cluster in d.columns and d[cluster].nunique() > 1:
        cov_type = "cluster"
        cov_kw = {"groups": d[cluster]}

    if model_type == "OLS":
        res = smf.ols(formula, data=d).fit(cov_type=cov_type, cov_kwds=cov_kw if cov_kw else None)
    elif model_type == "Poisson GLM":
        res = smf.glm(formula, data=d, family=sm.families.Poisson()).fit(cov_type=cov_type, cov_kwds=cov_kw if cov_kw else None)
    elif model_type == "Negative Binomial GLM":
        res = smf.glm(formula, data=d, family=sm.families.NegativeBinomial(alpha=1.0)).fit(cov_type=cov_type, cov_kwds=cov_kw if cov_kw else None)
    elif model_type == "Fractional logit / Binomial GLM":
        d["__y__"] = d[y].clip(0, 1)
        formula = make_formula("__y__", x_formula, cats, fe_region, fe_year)
        res = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(cov_type=cov_type, cov_kwds=cov_kw if cov_kw else None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return res, d, formula


def result_table(res) -> pd.DataFrame:
    params = res.params
    out = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "std_error": res.bse,
        "stat": getattr(res, "tvalues", getattr(res, "zvalues", np.nan)),
        "p_value": res.pvalues,
    })
    ci = res.conf_int()
    out["ci_low"] = ci.iloc[:, 0].values
    out["ci_high"] = ci.iloc[:, 1].values
    out["signif"] = pd.cut(out["p_value"], bins=[-0.001, 0.001, 0.01, 0.05, 0.1, 1], labels=["***", "**", "*", ".", ""])
    return out.sort_values("p_value")


def vif_table(d: pd.DataFrame, x: List[str]) -> pd.DataFrame:
    xs = [v for v in x if v in d.columns and pd.api.types.is_numeric_dtype(d[v])]
    xs = [v for v in xs if d[v].nunique(dropna=True) > 1]
    if len(xs) < 2:
        return pd.DataFrame({"message": ["At least two non-constant numeric regressors are needed for VIF."]})
    X = d[xs].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(X, has_constant="add")
    rows = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        try:
            rows.append({"variable": c, "VIF": variance_inflation_factor(X.values, i)})
        except Exception as e:
            rows.append({"variable": c, "VIF": np.nan, "error": str(e)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)

# -----------------------------
# GIS and spatial functions
# -----------------------------
# This GIS layer deliberately avoids geopandas/shapely/fiona/pyproj, because those
# compiled GIS dependencies often crash on older macOS/Python setups. It uses the
# official Eurostat GISCO NUTS-2 GeoJSON directly through requests + Plotly.

@st.cache_data(show_spinner=False)
def load_gisco_nuts2_geojson() -> Tuple[dict | None, str | None]:
    try:
        import requests
        r = requests.get(GISCO_NUTS2_GEOJSON, timeout=30)
        r.raise_for_status()
        gj = r.json()
        features = []
        for f in gj.get("features", []):
            props = f.get("properties", {})
            if props.get("CNTR_CODE") == "EL" and int(props.get("LEVL_CODE", 2)) == 2:
                features.append(f)
        if not features:
            return None, "No Greek NUTS-2 features found in GISCO GeoJSON."
        return {"type": "FeatureCollection", "features": features}, None
    except Exception as e:
        return None, str(e)


def _aggregate_for_map(panel_metric: pd.DataFrame, metric: str, year: int | None = None) -> pd.DataFrame:
    lookup = load_region_lookup()
    if year is not None and "year" in panel_metric.columns:
        data = panel_metric[panel_metric["year"] == year].copy()
    else:
        # For count/flow outcomes, sum across years; for rates, take the mean.
        if metric in ["absorption_rate_budget_mean", "public_share_of_budget"]:
            data = panel_metric.groupby(["region", "nuts_id", "region_el", "lat", "lon"], as_index=False)[metric].mean()
        else:
            data = panel_metric.groupby(["region", "nuts_id", "region_el", "lat", "lon"], as_index=False)[metric].sum()
    data = lookup.merge(data[["nuts_id", metric]] if metric in data.columns else pd.DataFrame(columns=["nuts_id", metric]), on="nuts_id", how="left")
    # Plotly map size/colour validators do not tolerate NaN/object values reliably.
    # The full lookup deliberately includes all 13 Greek regions, while the uploaded
    # dataset contains observations for 12 regions. Missing regional observations are
    # therefore kept on the map as zero-valued areas rather than crashing the app.
    if metric in data.columns:
        data[metric] = pd.to_numeric(data[metric], errors="coerce").fillna(0)
    return data


def plot_region_map(panel_metric: pd.DataFrame, metric: str, year: int | None = None):
    data = _aggregate_for_map(panel_metric, metric, year)
    geojson, err = load_gisco_nuts2_geojson()
    if geojson is not None:
        fig = px.choropleth_mapbox(
            data,
            geojson=geojson,
            locations="nuts_id",
            featureidkey="properties.NUTS_ID",
            color=metric,
            hover_name="region_el",
            hover_data={"nuts_id": True, metric: ":,.3f", "lat": False, "lon": False},
            mapbox_style="carto-positron",
            center={"lat": 38.6, "lon": 23.7},
            zoom=5.0,
            opacity=0.72,
            height=650,
        )
        fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0})
        return fig, None
    else:
        scatter_data = data.copy()
        if metric in scatter_data.columns:
            scatter_data[metric] = pd.to_numeric(scatter_data[metric], errors="coerce").fillna(0).clip(lower=0)
        fig = px.scatter_geo(
            scatter_data,
            lat="lat", lon="lon",
            size=metric if metric in scatter_data.columns else None,
            color=metric if metric in scatter_data.columns else None,
            hover_name="region_el", projection="natural earth", height=650,
            scope="europe",
        )
        fig.update_geos(center={"lat": 38.6, "lon": 23.7}, lataxis_range=[34, 42.5], lonaxis_range=[18, 30])
        return fig, f"Centroid fallback used. GISCO GeoJSON error: {err}"


def _knn_weights_from_latlon(df_: pd.DataFrame, k: int = 3) -> np.ndarray:
    coords = df_[["lat", "lon"]].astype(float).to_numpy()
    n = len(coords)
    k = max(1, min(k, n - 1))
    # Euclidean lat/lon is sufficient here for a compact Greek NUTS-2 exploratory diagnostic.
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dist[i])[:k]
        W[i, nn] = 1.0
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums[:, None]


def _global_moran(y: np.ndarray, W: np.ndarray) -> float:
    z = y - np.nanmean(y)
    denom = float(np.sum(z ** 2))
    if denom == 0:
        return np.nan
    S0 = float(W.sum())
    n = len(y)
    return (n / S0) * float(z @ W @ z) / denom


def spatial_diagnostics(panel_metric: pd.DataFrame, metric: str, year: int | None = None) -> pd.DataFrame:
    try:
        data = _aggregate_for_map(panel_metric, metric, year)
        data = data.dropna(subset=["lat", "lon", metric]).copy()
        if len(data) < 4:
            return pd.DataFrame({"diagnostic": ["Spatial diagnostics unavailable"], "detail": ["Fewer than four mapped regions with non-missing values."]})
        y = data[metric].astype(float).to_numpy()
        W = _knn_weights_from_latlon(data, k=3)
        I = _global_moran(y, W)
        # Small permutation test; enough for an exploratory dashboard and no compiled GIS dependencies.
        rng = np.random.default_rng(42)
        sims = np.array([_global_moran(rng.permutation(y), W) for _ in range(499)])
        p_sim = (np.sum(np.abs(sims) >= abs(I)) + 1) / (len(sims) + 1) if np.isfinite(I) else np.nan
        z = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) != 0 else 1)
        wz = W @ z
        labels = []
        for zi, wi in zip(z, wz):
            if zi >= 0 and wi >= 0:
                labels.append("High-High")
            elif zi < 0 and wi < 0:
                labels.append("Low-Low")
            elif zi >= 0 and wi < 0:
                labels.append("High-Low")
            else:
                labels.append("Low-High")
        head = pd.DataFrame({
            "diagnostic": ["Global Moran's I", "Permutation p-value", "Weights"],
            "value": [I, p_sim, "3-nearest-neighbour, row-standardised"],
        })
        clusters = data[["nuts_id", "region_el", metric]].copy()
        clusters["LISA_cluster_proxy"] = labels
        clusters["spatial_lag_z"] = wz
        return pd.concat([head, clusters], ignore_index=True, sort=False)
    except Exception as e:
        return pd.DataFrame({"diagnostic": ["Spatial diagnostics error"], "detail": [str(e)]})

# -----------------------------
# UI
# -----------------------------
df = load_data()
catalog = load_catalog()
model_specs = load_model_specs()

st.title("R&D Projects, Regional Development and Econometric Decision Dashboard")
st.caption("Python-first dashboard with optional R replication, panel econometrics, project-level models, GIS choropleths and GAMS-style scenario analysis.")

with st.sidebar:
    st.header("Modules")
    page = st.radio(
        "Choose view",
        ["1. Data audit", "2. Variable dictionary", "3. Regional panel", "4. Econometric models", "5. GIS & spatial diagnostics", "6. Scenario engine"],
        index=0,
    )
    st.divider()
    st.write("Dataset")
    st.write(f"Rows: **{df.shape[0]:,}**")
    st.write(f"Columns: **{df.shape[1]:,}**")
    if "region" in df.columns:
        st.write(f"Regions: **{df['region'].nunique()}**")

if page == "1. Data audit":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", f"{df.shape[0]:,}")
    c2.metric("Variables", f"{df.shape[1]:,}")
    c3.metric("Regions", f"{df['region'].nunique() if 'region' in df else 0}")
    c4.metric("Start years", f"{int(df['project_start_year'].min())}–{int(df['project_start_year'].max())}")

    st.subheader("Project distribution")
    left, right = st.columns(2)
    with left:
        if "region" in df:
            reg = df.groupby("region").size().reset_index(name="projects").sort_values("projects", ascending=False)
            st.plotly_chart(px.bar(reg, x="projects", y="region", orientation="h", title="Projects by region"), use_container_width=True)
    with right:
        if "project_start_year" in df:
            yr = df.groupby("project_start_year").size().reset_index(name="projects")
            st.plotly_chart(px.line(yr, x="project_start_year", y="projects", markers=True, title="Projects by start year"), use_container_width=True)

    st.subheader("Missingness and types")
    miss = catalog[["clean_variable", "thematic_group", "python_dtype", "missing_n", "missing_pct", "unique_n"]].sort_values("missing_pct", ascending=False)
    st.dataframe(miss, use_container_width=True, hide_index=True)
    st.subheader("Raw preview")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "2. Variable dictionary":
    st.subheader("All identified variables")
    groups = ["All"] + sorted(catalog["thematic_group"].dropna().unique())
    group = st.selectbox("Filter by thematic block", groups)
    show = catalog.copy()
    if group != "All":
        show = show[show["thematic_group"] == group]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Research-question model specification")
    st.dataframe(model_specs, use_container_width=True, hide_index=True)

elif page == "3. Regional panel":
    mode = st.selectbox("Panel construction", ["Start-year allocation", "End-year allocation", "Active years, even allocation"])
    panel = build_region_year_panel(df, mode)
    st.write(f"Panel rows: **{len(panel):,}** | Regions: **{panel['region'].nunique()}** | Years: **{int(panel['year'].min())}–{int(panel['year'].max())}**")
    st.dataframe(panel, use_container_width=True, hide_index=True)

    metrics = [c for c in ["project_count", "absorption_rate_budget_mean", "collaborative_projects", "patents", "spin_offs", "smes_benefited", "participating_firms", "total_budget_allocated", "total_public_expenditure_allocated"] if c in panel.columns]
    metric = st.selectbox("Metric for regional trend", metrics)
    trend = panel.groupby("year", as_index=False)[metric].sum() if metric != "absorption_rate_budget_mean" else panel.groupby("year", as_index=False)[metric].mean()
    st.plotly_chart(px.line(trend, x="year", y=metric, markers=True, title=f"Regional panel trend: {metric}"), use_container_width=True)

elif page == "4. Econometric models":
    level = st.radio("Level of analysis", ["Project-level", "Region-year panel"], horizontal=True)
    if level == "Region-year panel":
        mode = st.selectbox("Panel construction", ["Start-year allocation", "End-year allocation", "Active years, even allocation"], key="model_panel_mode")
        data = build_region_year_panel(df, mode)
        default_y = "absorption_rate_budget_mean"
        y_options = [c for c in numeric_cols(data) if c not in ["lat", "lon", "year"]]
        cats = []
        cluster_options = [None, "region", "year"]
    else:
        data = df.copy()
        default_y = "indicator_6_num_of_research_job_ipa"
        y_options = [c for c in numeric_cols(data) if c not in ["lat", "lon"]]
        cats = st.multiselect("Categorical controls", available(PROJECT_CONTROLS_CAT, data), default=["rtdi_sector", "scientific_field", "type_of_beneficiary", "region"])
        cluster_options = [None, "region", "project_start_year"]

    st.subheader("Model setup")
    y = st.selectbox("Dependent variable", y_options, index=y_options.index(default_y) if default_y in y_options else 0)
    block_names = st.multiselect("Regressor blocks", list(BLOCKS.keys()), default=["Economic development / business base", "R&D human capital / institutions"])
    candidate_x = []
    for b in block_names:
        candidate_x.extend(available(BLOCKS[b], data))
    candidate_x = list(dict.fromkeys(candidate_x))
    x = st.multiselect("Regressors", candidate_x, default=candidate_x[: min(6, len(candidate_x))])
    model_type = st.selectbox("Estimator", ["OLS", "Poisson GLM", "Negative Binomial GLM", "Fractional logit / Binomial GLM"])
    transform_y = st.selectbox("Y transformation", ["none", "log1p", "fractional_clip"], index=1 if model_type == "OLS" else 0)
    transform_x = st.checkbox("Use log1p transform for non-negative regressors", value=True)
    fe_region = st.checkbox("Region fixed effects", value=(level == "Region-year panel"))
    fe_year = st.checkbox("Year fixed effects", value=(level == "Region-year panel")) if "year" in data.columns else False
    cluster = st.selectbox("Cluster-robust standard errors", cluster_options, index=1 if "region" in cluster_options else 0)

    if st.button("Run model", type="primary"):
        try:
            res, model_data, formula = fit_model(data, y, x, model_type, transform_y, transform_x, cats, fe_region, fe_year, cluster)
            st.code(formula, language="text")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("N", f"{int(res.nobs):,}")
            mc2.metric("AIC", fmt_num(getattr(res, "aic", np.nan)))
            mc3.metric("BIC", fmt_num(getattr(res, "bic", np.nan)))
            mc4.metric("R² / pseudo", fmt_num(getattr(res, "rsquared", np.nan)))

            rt = result_table(res)
            st.subheader("Coefficient table")
            st.dataframe(rt, use_container_width=True, hide_index=True)
            st.download_button("Download coefficient table", rt.to_csv(index=False).encode("utf-8-sig"), "model_coefficients.csv")

            st.subheader("Diagnostics")
            left, right = st.columns(2)
            with left:
                st.write("Variance inflation factors")
                st.dataframe(vif_table(model_data, x), use_container_width=True, hide_index=True)
            with right:
                pred = res.predict(model_data)
                actual = model_data[y] if y in model_data else np.nan
                plot_df = pd.DataFrame({"actual": actual, "predicted": pred})
                st.plotly_chart(px.scatter(plot_df, x="predicted", y="actual", trendline="ols", title="Actual vs fitted"), use_container_width=True)

            with st.expander("Full statsmodels summary"):
                st.text(res.summary().as_text())
        except Exception as e:
            st.error(f"Model failed: {e}")
            st.info("Reduce the number of regressors, remove high-cardinality dummies, or switch to OLS/log1p for sparse count outcomes.")

elif page == "5. GIS & spatial diagnostics":
    mode = st.selectbox("Panel construction", ["Start-year allocation", "End-year allocation", "Active years, even allocation"], key="gis_panel_mode")
    panel = build_region_year_panel(df, mode)
    metric_options = [c for c in ["project_count", "absorption_rate_budget_mean", "collaborative_projects", "patents", "spin_offs", "smes_benefited", "participating_firms", "total_budget_allocated", "total_public_expenditure_allocated"] if c in panel.columns]
    metric = st.selectbox("Map metric", metric_options)
    year_mode = st.radio("Map time aggregation", ["All years average/sum", "Specific year"], horizontal=True)
    year = None
    if year_mode == "Specific year":
        year = int(st.selectbox("Year", sorted(panel["year"].dropna().astype(int).unique())))
    fig, warning = plot_region_map(panel, metric, year)
    if warning:
        st.warning(warning)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spatial autocorrelation / LISA diagnostics")
    sd = spatial_diagnostics(panel, metric, year)
    st.dataframe(sd, use_container_width=True, hide_index=True)
    st.caption("KNN spatial weights are used by default to avoid island-neighbour failures for the Greek archipelago. For a strict contiguity-only version, switch the code to Queen weights in spatial_diagnostics().")

elif page == "6. Scenario engine":
    st.subheader("GAMS-style scenario layer")
    st.write("This module estimates a reduced-form model and applies a regional shock to one driver. It is not an optimisation solver; it is a decision-support scenario layer that can later be replaced by a true GAMS/Pyomo optimisation block.")
    mode = st.selectbox("Panel construction", ["Start-year allocation", "End-year allocation", "Active years, even allocation"], key="sc_panel_mode")
    panel = build_region_year_panel(df, mode)
    outcome = st.selectbox("Outcome", [c for c in ["project_count", "absorption_rate_budget_mean", "collaborative_projects", "patents", "spin_offs", "smes_benefited", "participating_firms"] if c in panel.columns])
    driver_candidates = available(ECONOMIC_BLOCK + RD_HUMAN_BLOCK + RD_EXP_BLOCK + INNOVATION_BLOCK + SCIENCE_BLOCK + ["mean_budget", "mean_public_expenditure"], panel)
    driver = st.selectbox("Shock variable", driver_candidates)
    shock_pct = st.slider("Shock (%)", min_value=-50, max_value=100, value=10, step=5) / 100
    selected_region = st.selectbox("Apply to", ["All regions"] + sorted(panel["region"].dropna().unique()))
    control_options = [c for c in driver_candidates if c != driver]
    default_controls = control_options[:3]
    controls = st.multiselect("Additional controls", control_options, default=default_controls)
    estimator = "OLS" if outcome == "absorption_rate_budget_mean" else "Poisson GLM"

    if st.button("Run scenario", type="primary"):
        try:
            x = [driver] + [c for c in controls if c != driver]
            res, model_data, formula = fit_model(panel, outcome, x, estimator, "log1p" if estimator == "OLS" else "none", True, [], True, True, "region")
            baseline = model_data.copy()
            shocked = model_data.copy()
            mask = np.ones(len(shocked), dtype=bool) if selected_region == "All regions" else shocked["region"].eq(selected_region).values
            shocked.loc[mask, driver] = shocked.loc[mask, driver] * (1 + shock_pct)
            # Recreate log columns if needed.
            shocked, _ = add_log_columns(shocked, x)
            pred0 = res.predict(baseline)
            pred1 = res.predict(shocked)
            out = baseline[["region", "year", outcome, driver]].copy()
            out["baseline_prediction"] = pred0
            out["scenario_prediction"] = pred1
            out["delta"] = out["scenario_prediction"] - out["baseline_prediction"]
            out["delta_pct"] = out["delta"] / out["baseline_prediction"].replace(0, np.nan)
            st.code(formula, language="text")
            st.dataframe(out.sort_values("delta", ascending=False), use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(out.groupby("region", as_index=False)["delta"].mean().sort_values("delta"), x="delta", y="region", orientation="h", title="Mean scenario effect by region"), use_container_width=True)
            st.download_button("Download scenario results", out.to_csv(index=False).encode("utf-8-sig"), "scenario_results.csv")
        except Exception as e:
            st.error(f"Scenario failed: {e}")
