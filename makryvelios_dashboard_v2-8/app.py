"""Makryvelios Research Analytics & Econometrics Workbench v5.8.4.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog

from analytics_core import (
    MAX_DEPENDENT, MAX_INDEPENDENT, categorical_summary, chi_square_tests,
    cluster_table, combine_frames, correlation_matrix, data_dictionary,
    descriptive_statistics, fit_detailed_model, group_tests,
    matrix_ols_many_outcomes, normality_tests, p_adjust, pca_table,
    quality_summary, read_tabular_bytes, serialisable_summary, tidy_frame,
    time_series_tests, to_excel_bytes, vif_table, regularised_regression,
    instrumental_variables_2sls, difference_in_differences, granger_table,
    arima_forecast, cronbach_alpha, monte_carlo_ols, monte_carlo_portfolio,
    outlier_summary, is_likely_analytical_frame,
)
from legacy_rd import build_region_year_panel, is_rd_dataset, original_model_presets, rd_column_blocks
from mapping import (
    GISCO_URLS, aggregate_geography, choropleth_figure, fetch_geojson,
    map_commentary, moran_diagnostics, static_map_bytes,
)
from reporting import build_html_report
from advanced_analytics import advanced_clustering, panel_model_suite, predictive_model_comparison
from mcda import METHODS as MCDA_METHODS, WEIGHT_METHODS, mcda_analysis, mcda_publication_bundle
from ita_ui import render_ita_module
from gams_ui import render_gams_studio
from respondent_ui import render_respondent_module
from frontier_ui import render_frontier_methods
from agentic_ui import render_agentic_research
from visuals import (
    interactive_figure, publication_bundle, ols_publication_bundle,
    monte_carlo_publication_bundle, clustering_publication_bundle,
    predictive_publication_bundle, panel_publication_bundle,
)
from output_guidance import explain_output
from prompt_library import prompt_library, prompt_library_markdown
from research_chair import (
    add_safe_derived_column, apply_scope,
    build_paper_blueprint, docx_bytes, execute_protocol,
    execute_natural_language_command, execute_question_batch,
    extract_pdf_collection, ollama_models, ollama_reply,
    research_bundle, select_pdf_evidence, year_bounds, chair_interactive_figures,
)
from llm_bridge import configured as llm_configured, llm_reply


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RQS = pd.read_csv(BASE / "research_questions.csv")
HYPOTHESES = pd.read_csv(BASE / "research_hypotheses.csv")
EVIDENCE = pd.read_csv(BASE / "source_evidence_catalogue.csv")

st.set_page_config(page_title="Makryvelios Research Analytics", page_icon="🇬🇷", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#20D5E6", "#D9A441", "#A57CFF", "#FF6B8A", "#54D68B", "#77A8FF"]
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 80% -10%,#0b7f9b38 0,transparent 34%),radial-gradient(circle at 10% 30%,#6a3fb52c 0,transparent 30%),linear-gradient(180deg,#102131 0%,#142b3d 48%,#0f2233 100%);color:#eaf7fb}
.block-container{padding-top:1.2rem;padding-bottom:4rem;max-width:1550px}
.hero{position:relative;overflow:hidden;background:linear-gradient(118deg,#133149dd 0%,#16465ddd 48%,#0f6076dd 100%);color:white;padding:1.8rem 1.95rem;border:1px solid #38e7f65c;border-radius:20px;margin-bottom:1.15rem;box-shadow:0 20px 70px #00000066,inset 0 1px 0 #ffffff12;backdrop-filter:blur(12px)}
.hero:before{content:"";position:absolute;inset:0;background-image:linear-gradient(#2fe4f014 1px,transparent 1px),linear-gradient(90deg,#2fe4f014 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(90deg,#000,transparent)}
.hero:after{content:"";position:absolute;width:360px;height:360px;right:-90px;top:-220px;border:1px solid #67efff75;border-radius:50%;box-shadow:0 0 90px #21d4fd45,inset 0 0 45px #21d4fd20}
.hero h1{font-size:2.05rem;letter-spacing:-.025em;margin:0 0 .4rem}.hero p{margin:0;opacity:.88;max-width:980px}.status{display:inline-block;margin-bottom:.65rem;padding:.22rem .62rem;border:1px solid #56e6f4;border-radius:999px;color:#9af5ff;font-size:.72rem;font-weight:700;letter-spacing:.12em}
.chip{position:relative;display:inline-block;margin:.8rem .4rem 0 0;padding:.28rem .68rem;background:#ffffff14;border:1px solid #52e8f43a;border-radius:8px;font-size:.77rem;color:#dffcff;box-shadow:inset 0 0 14px #2edbea0d}
.guide{background:linear-gradient(105deg,#14354add,#183f55dd);border:1px solid #1b7382;border-left:5px solid #20d5e6;border-radius:12px;padding:1rem 1.1rem;margin:.4rem 0 1.1rem;color:#dcecf2;box-shadow:0 8px 26px #0000002c}.guide b{color:#55ebf7}
.small-note{font-size:.86rem;color:#91a8b5}.stDataFrame{border:1px solid #173d50;border-radius:12px;box-shadow:0 8px 28px #00000030;overflow:hidden}
div[data-testid="stMetric"]{border:1px solid #164357;border-radius:14px;padding:.78rem;background:linear-gradient(145deg,#15364a,#102d40);box-shadow:0 10px 28px #00000038,inset 0 1px 0 #ffffff09}
div[data-testid="stMetricValue"]{color:#57e8f4;text-shadow:0 0 18px #20d5e64f}div[data-testid="stMetricLabel"]{color:#a8bbc5}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #168297;background:#12374a;color:#dffbff;transition:all .18s ease;box-shadow:0 4px 14px #00000030}.stButton>button:hover,.stDownloadButton>button:hover{border-color:#4de9f5;color:white;box-shadow:0 0 22px #20d5e638;transform:translateY(-1px)}.stButton>button[kind="primary"]{background:linear-gradient(110deg,#08738a,#12a8b5);border:1px solid #4de9f5;box-shadow:0 8px 24px #08788c55}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e2133 0%,#143449 55%,#102c40 100%);border-right:1px solid #143448;box-shadow:8px 0 30px #00000040}section[data-testid="stSidebar"] *{color:#e8f7fb}section[data-testid="stSidebar"] [data-baseweb="select"]>div,section[data-testid="stSidebar"] input{background:#17384d!important;color:#eaf7fb!important}
/* White narrative text on dark surfaces; light input controls remain unchanged. */
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li,.stApp [data-testid="stMarkdownContainer"] a,.stApp h4,.stApp h5,.stApp h6{color:#FFFFFF!important}
/* High-contrast widget labels on every dark panel. Input values remain untouched. */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,.stSelectbox label p,.stMultiSelect label p,.stNumberInput label p,.stTextInput label p,.stTextArea label p,.stDateInput label p,.stTimeInput label p,.stSlider label p,.stCheckbox label p,.stRadio label p,.stToggle label p,.stFileUploader label p{color:#D8C7FF!important;opacity:1!important;font-weight:600!important}
[data-testid="stTooltipIcon"]{color:#CBB8FF!important;opacity:1!important}
[data-testid="stFileUploader"]{background:linear-gradient(145deg,#14384d,#18465b);border:1px solid #1b6979;border-radius:14px;padding:.65rem;box-shadow:inset 0 0 24px #20d5e609,0 8px 24px #00000035}
[data-testid="stFileUploader"] label{color:#9ef5ff!important;font-weight:700!important;letter-spacing:.01em}
[data-testid="stFileUploaderDropzone"]{background:linear-gradient(135deg,#173f54,#123449)!important;border:1.5px dashed #26cddd!important;border-radius:12px!important;color:#eafcff!important;min-height:116px}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploader"] button{background:linear-gradient(110deg,#0b8298,#19b7c2)!important;color:#ffffff!important;border:1px solid #65f3ff!important;border-radius:9px!important;font-weight:800!important;box-shadow:0 0 18px #20d5e63b!important}
[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzoneInstructions"]{color:#a9c5d0!important}
/* Uploaded-file cards are light: filenames and sizes must be genuinely dark. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"]{background:#f4f7fa!important;border:1px solid #d3dce5!important;border-radius:10px!important;color:#101820!important;-webkit-text-fill-color:#101820!important}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] *,section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"],section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"] *{color:#101820!important;-webkit-text-fill-color:#101820!important;opacity:1!important}
/* Typed and selected text on every light input surface: force true black. */
.stApp input,.stApp textarea,.stApp [contenteditable="true"],.stTextInput input,.stTextArea textarea,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{color:#000000!important;-webkit-text-fill-color:#000000!important;caret-color:#000000!important;color-scheme:light!important}
.stApp input::placeholder,.stApp textarea::placeholder,.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:#59616d!important;-webkit-text-fill-color:#59616d!important;opacity:1!important}
.stApp input::selection,.stApp textarea::selection,.stApp [contenteditable="true"]::selection{color:#000000!important;-webkit-text-fill-color:#000000!important;background:#a9d8ff!important}
/* Prepared/editable prompts: white surface and black type, including code-style prompt previews. */
.stApp [data-testid="stTextArea"]>div,.stApp [data-testid="stTextInput"]>div{background:#ffffff!important;border-radius:10px!important}
.stApp [data-testid="stTextArea"] textarea,.stApp [data-testid="stTextInput"] input{background:#ffffff!important;color:#000000!important;-webkit-text-fill-color:#000000!important}
.stApp [data-testid="stCodeBlock"],.stApp [data-testid="stCodeBlock"] pre,.stApp [data-testid="stCodeBlock"] code,.stApp pre,.stApp pre code{background:#ffffff!important;color:#000000!important;-webkit-text-fill-color:#000000!important;border-color:#cbd5df!important}
.stApp [data-testid="stCodeBlock"] button{color:#111111!important;background:#eef2f5!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:#dce2e8!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"],[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] *,[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"],[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] *{color:#101318!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea{color:#eaf7fb!important;-webkit-text-fill-color:#eaf7fb!important;caret-color:#eaf7fb!important}
[data-baseweb="tab-list"]{gap:.4rem;background:#123449;border:1px solid #12384b;border-radius:12px;padding:.35rem}[data-baseweb="tab"]{border-radius:8px;padding:.5rem .8rem;color:#FFFFFF!important}[data-baseweb="tab"] *{color:#FFFFFF!important}[aria-selected="true"]{background:#176077!important;color:#FFFFFF!important}
[data-testid="stExpander"]{background:#123449;border:1px solid #15384a;border-radius:12px}.stAlert{border-radius:12px;border:1px solid #1b5265;background:#15384c}.stAlert,.stAlert *,[data-testid="stAlert"],[data-testid="stAlert"] *,[data-baseweb="notification"],[data-baseweb="notification"] *{color:#FFFFFF!important}
h1,h2,h3{letter-spacing:-.018em;color:#effcff}h2{border-bottom:1px solid #173849;padding-bottom:.45rem}hr{border-color:#183344}.download-row{margin-top:.2rem}
code{color:#7df3ff;background:#14384a!important}.stCaptionContainer,.stCaptionContainer p,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:#D7E7EE!important;-webkit-text-fill-color:#D7E7EE!important;opacity:1!important}
@media(max-width:760px){.hero{padding:1.25rem}.hero h1{font-size:1.55rem}.block-container{padding-left:.8rem;padding-right:.8rem}}

/* v5.7.2 contrast guard: light controls always use dark text, dark surfaces always use light text. */
section[data-testid="stSidebar"] [data-testid="stTextInput"]>div,section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#FFFFFF!important;border:1px solid #94A3B8!important}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input,section[data-testid="stSidebar"] [data-testid="stTextInput"] input[type="password"],section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *{color:#0F172A!important;-webkit-text-fill-color:#0F172A!important}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder{color:#475569!important;-webkit-text-fill-color:#475569!important}
[data-testid="stAlert"]{background:#102D40!important;border:1px solid #2D6078!important;color:#FFFFFF!important}
[data-testid="stAlert"] *{color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important}
[data-testid="stDataFrame"]{background:#FFFFFF!important;color:#111827!important}
.llm-panel{padding:.8rem .85rem;margin:.6rem 0;border:1px solid #D946EF;border-left:5px solid #D946EF;border-radius:11px;background:#1E293B;color:#FFFFFF;box-shadow:0 8px 24px #00000035}
.llm-panel b{color:#F5D0FE!important}.llm-panel small{color:#E2E8F0!important}
.gams-badge{display:inline-block;padding:.24rem .55rem;border-radius:999px;background:#0B1F3A;border:1px solid #38BDF8;color:#F8FAFC!important;font-weight:800;font-size:.78rem}

</style>
<div class="hero"><span class="status">POSTDOCTORAL ANALYTICAL ENGINE v5.8.4 · ALL v5.7.2 + EARLIER CAPABILITIES RETAINED</span><h1>Makryvelios Research Analytics &amp; Econometrics Command Centre</h1><p>R&amp;D projects • «Αντώνης Τρίτσης» • renewable-energy portfolios • causal and predictive econometrics • Monte Carlo • advanced clustering • MCDA • ITA public-funding optimisation • respondent-level expert analytics • panel models • Greece spatial intelligence • publication systems • causal/Bayesian/XAI/Pareto frontier methods • agentic research automation</p><span class="chip">One-screen Research Chair</span><span class="chip">ITA-PB + Hybrid ITA-RW</span><span class="chip">Empirical respondent weights</span><span class="chip">Visible GAMS-compatible Studio</span><span class="chip">GAMS-ready export</span><span class="chip">User-key LLM co-pilot</span><span class="chip">52 editable prompts</span><span class="chip">Batch genuine answers</span><span class="chip">Feasibility verdicts</span><span class="chip">Interactive HTML</span><span class="chip">600-dpi + vector output</span><span class="chip">Analytical-sheet detection</span><span class="chip">No-key basemap</span><span class="chip">Offline high-detail Greece maps</span><span class="chip">Readable uploaded filenames</span><span class="chip">Frontier Methods Lab</span><span class="chip">Agentic Research Mode</span><span class="chip">150-RQ batches</span><span class="chip">Offline submission bundles</span><span class="chip">Original menu retained</span></div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def parse_payloads(items: tuple[tuple[str, bytes], ...], all_sheets: bool, normalise: bool) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, payload in items:
        for label, frame in read_tabular_bytes(name, payload, all_sheets=all_sheets).items():
            frames[label] = tidy_frame(frame, normalise_columns=normalise)
    return frames


@st.cache_data(show_spinner=False)
def parse_pdf_payloads(items: tuple[tuple[str, bytes], ...]) -> pd.DataFrame:
    return extract_pdf_collection(items)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_geojson(level: str, custom: bytes | None = None) -> dict:
    return fetch_geojson(level, custom)


def load_bundled() -> dict[str, pd.DataFrame]:
    candidates = list(DATA.glob("*.xlsx")) + list(DATA.glob("*.csv"))
    frames: dict[str, pd.DataFrame] = {}
    for path in candidates:
        try:
            for label, frame in read_tabular_bytes(path.name, path.read_bytes(), all_sheets=False).items():
                frames[label] = tidy_frame(frame, normalise_columns=True)
        except Exception:
            continue
    return frames


def download_table(label: str, table: pd.DataFrame, filename: str) -> None:
    st.download_button(label, table.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv")


def module_guide(purpose: str, steps: str, interpretation: str) -> None:
    st.markdown(
        f'<div class="guide"><b>Purpose.</b> {purpose}<br><b>How to use it.</b> {steps}<br><b>Interpretation.</b> {interpretation}</div>',
        unsafe_allow_html=True,
    )


def table_with_downloads(title: str, table: pd.DataFrame, stem: str, explanation: str | None = None, max_rows: int = 10_000) -> None:
    st.subheader(title)
    if explanation:
        st.caption(explanation)
    guide = explain_output(title, table, "table")
    st.caption("Plain-language meaning: " + guide["plain_language_summary"])
    st.dataframe(table.head(max_rows), width="stretch", hide_index=True)
    with st.expander(f"How to read and use: {title}", expanded=False):
        st.markdown(
            f"**What it is.** {guide['what_it_is']}\n\n"
            f"**Why it is used.** {guide['why_it_is_used']}\n\n"
            f"**How to read it.** {guide['how_to_read_it']}\n\n"
            f"**What the pattern means.** {guide['what_the_pattern_means']}\n\n"
            f"**What it does not mean.** {guide['what_it_does_not_mean']}\n\n"
            f"**What to do next.** {guide['recommended_next_step']}"
        )
    guide_table = pd.DataFrame([guide])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(f"Download {title} (CSV)", table.to_csv(index=False).encode("utf-8-sig"), f"{stem}.csv", "text/csv", key=f"csv_{stem}")
    with c2:
        st.download_button(f"Download {title} (Excel)", to_excel_bytes({title: table, "Output guide": guide_table}), f"{stem}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{stem}")
    with c3:
        st.download_button("Download explanation (CSV)", guide_table.to_csv(index=False).encode("utf-8-sig"), f"{stem}_guide.csv", "text/csv", key=f"guide_{stem}")


def guided_dataframe(title: str, table: pd.DataFrame, max_rows: int = 10_000, hide_index: bool = True) -> None:
    """Display an existing table without changing its workflow, plus an interpretation guide."""
    st.dataframe(table.head(max_rows), width="stretch", hide_index=hide_index)
    guide = explain_output(title, table, "table")
    st.caption("Plain-language meaning: " + guide["plain_language_summary"])
    with st.expander(f"How to read and use: {title}", expanded=False):
        st.markdown(
            f"**What it is.** {guide['what_it_is']}\n\n"
            f"**Why it is used.** {guide['why_it_is_used']}\n\n"
            f"**How to read it.** {guide['how_to_read_it']}\n\n"
            f"**What the pattern means.** {guide['what_the_pattern_means']}\n\n"
            f"**What it does not mean.** {guide['what_it_does_not_mean']}\n\n"
            f"**What to do next.** {guide['recommended_next_step']}"
        )


def figure_with_downloads(fig, stem: str, data: pd.DataFrame | None = None, explanation: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch")
    if explanation:
        st.info(explanation)
    guide = explain_output(stem, data, "chart")
    st.caption("Plain-language meaning: " + guide["plain_language_summary"])
    with st.expander("How to read and use this chart", expanded=False):
        st.markdown(
            f"**What it is.** {guide['what_it_is']}\n\n"
            f"**Why it is used.** {guide['why_it_is_used']}\n\n"
            f"**How to read it.** {guide['how_to_read_it']}\n\n"
            f"**What the pattern means.** {guide['what_the_pattern_means']}\n\n"
            f"**What it does not mean.** {guide['what_it_does_not_mean']}\n\n"
            f"**What to do next.** {guide['recommended_next_step']}"
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download interactive figure (HTML)", fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8"), f"{stem}.html", "text/html", key=f"html_{stem}")
    with c2:
        if data is not None:
            st.download_button("Download plotted data (CSV)", data.to_csv(index=False).encode("utf-8-sig"), f"{stem}_data.csv", "text/csv", key=f"data_{stem}")
    with c3:
        st.download_button("Download chart explanation (CSV)", pd.DataFrame([guide]).to_csv(index=False).encode("utf-8-sig"), f"{stem}_guide.csv", "text/csv", key=f"figure_guide_{stem}")


with st.sidebar:
    st.header("Data intake console")
    st.markdown('<div style="padding:.55rem .7rem;margin:0 0 .5rem;border:1px solid #1d7282;border-radius:9px;background:#12374b;color:#8cf4ff;font-size:.78rem;font-weight:800;letter-spacing:.06em">⬆ SECURE MULTI-FILE UPLOAD</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload one or many XLSX/XLS/CSV/TSV/Parquet/Arrow files", type=["xlsx", "xlsm", "xls", "csv", "tsv", "parquet", "pq", "feather", "arrow"], accept_multiple_files=True)
    all_sheets = st.checkbox("Read every Excel sheet", value=True)
    normalise = st.checkbox("Normalise variable names", value=True, help="Recommended for modelling; original values are not altered.")
    if uploaded:
        payloads = tuple((f.name, f.getvalue()) for f in uploaded)
        try:
            frames = parse_payloads(payloads, all_sheets, normalise)
        except Exception as exc:
            st.error(f"Upload could not be read: {exc}")
            frames = {}
    else:
        frames = load_bundled()
        if frames:
            st.caption("Bundled R&D reference workbook loaded. Uploading files replaces it for this session.")

    if frames:
        relationship_options = [
            "Keep datasets separate",
            "Combine columns side-by-side (by row order)",
            "Append rows (union by column name)",
            "Join datasets on key(s)",
        ]
        mode = st.selectbox(
            "Dataset relationship",
            relationship_options,
            help="Choose whether to analyse one source, place all selected source columns side-by-side, append observations, or join sources by verified keys.",
        )
        if mode == "Keep datasets separate":
            selected_label = st.selectbox("Active dataset", list(frames))
            selected_frames = {selected_label: frames[selected_label]}
        else:
            analytical_defaults = [label for label, frame in frames.items() if is_likely_analytical_frame(label, frame)]
            documentation_labels = [label for label in frames if label not in analytical_defaults]
            selected_datasets = st.multiselect(
                "Datasets to combine",
                list(frames),
                default=analytical_defaults or list(frames),
                help="Probable observation tables are selected automatically. README, dictionary, codebook, crosswalk, metadata and processing sheets remain available but are excluded by default.",
            )
            selected_frames = {label: frames[label] for label in selected_datasets}
            selected_label = f"{mode}: {len(selected_frames)} dataset(s)"
            if documentation_labels:
                st.caption(f"Automatically excluded {len(documentation_labels)} documentation/lookup sheet(s). They remain available in the selector if genuinely required.")
            if not selected_frames:
                st.warning("Select at least one dataset to continue.")

        common = sorted(set.intersection(*(set(d.columns) for d in selected_frames.values()))) if len(selected_frames) > 1 else (list(next(iter(selected_frames.values())).columns) if selected_frames else [])
        join_keys = st.multiselect("Join key(s)", common) if mode == "Join datasets on key(s)" else []
        join_how = st.selectbox("Join type", ["outer", "left", "inner", "right"]) if join_keys else "outer"
        if mode == "Combine columns side-by-side (by row order)" and selected_frames:
            row_counts = {label: len(frame) for label, frame in selected_frames.items()}
            st.info("All columns from the selected datasets will be available together in every analytical module.")
            st.caption("Row-order rule: row 1 is aligned with row 1, row 2 with row 2, and so forth. Use this only when every source has the same observational order.")
            if len(set(row_counts.values())) > 1:
                st.warning("The selected analytical tables have unequal row counts. They must not be interpreted as matched observations unless row identity and ordering have been independently verified.")
                with st.expander("Show row counts and alignment details", expanded=False):
                    count_table = pd.DataFrame({"dataset": list(row_counts), "rows": list(row_counts.values())}).sort_values("rows", ascending=False)
                    st.dataframe(count_table, hide_index=True, width="stretch")
                    st.caption("Shorter tables receive missing cells after positional alignment. Prefer separate analyses or a verified-key/aggregate join when units differ.")
        try:
            if not selected_frames:
                df = pd.DataFrame()
            else:
                df = frames[selected_label] if mode == "Keep datasets separate" else combine_frames(selected_frames, mode, join_keys, join_how)
        except Exception as exc:
            st.error(str(exc))
            fallback_label = next(iter(frames))
            selected_label = fallback_label
            df = frames[fallback_label]
    else:
        selected_label = "No dataset"
        df = pd.DataFrame()

    st.markdown('<div class="llm-panel"><b>AI / LLM RESEARCH ENGINE</b><br><small>Optional. Numerical analysis and optimisation remain deterministic. Choose hosted AI, a free-plan route, or local Ollama.</small></div>', unsafe_allow_html=True)
    llm_provider = st.selectbox(
        "AI provider",
        [
            "Anthropic Claude",
            "Google Gemini — free tier available",
            "Groq — free plan available",
            "Ollama Local — no API key",
            "OpenAI-compatible",
        ],
        key="global_llm_provider",
    )
    provider_defaults = {
        "Anthropic Claude": ("claude-sonnet-5", ""),
        "Google Gemini — free tier available": ("gemini-3.7-flash", ""),
        "Groq — free plan available": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1"),
        "Ollama Local — no API key": ("llama3.1:8b", "http://127.0.0.1:11434"),
        "OpenAI-compatible": ("", "https://api.openai.com/v1"),
    }
    default_llm_model, default_base_url = provider_defaults[llm_provider]
    needs_key = not llm_provider.startswith("Ollama")
    llm_api_key = st.text_input(
        "API Key" if needs_key else "API Key",
        type="password",
        disabled=not needs_key,
        placeholder="Not required for local Ollama" if not needs_key else "Paste the provider key",
        key=f"global_llm_api_key_{llm_provider}",
        help="Session-only. It is never written to exports. Ollama Local does not require a key.",
    )
    llm_model = st.text_input("Model ID", value=default_llm_model, key=f"global_llm_model_{llm_provider}")
    llm_base_url = default_base_url
    if llm_provider in {"OpenAI-compatible", "Ollama Local — no API key", "Groq — free plan available"}:
        label = "Ollama endpoint" if llm_provider.startswith("Ollama") else "API base URL"
        llm_base_url = st.text_input(label, value=default_base_url, key=f"global_llm_base_url_{llm_provider}")
    if llm_provider.startswith("Google Gemini"):
        st.warning("Gemini's free tier may use submitted content to improve Google products. Do not send confidential/restricted research material unless that is acceptable for the project.")
    elif llm_provider.startswith("Groq"):
        st.caption("Groq provides a free plan with rate limits. It is useful for fast synthesis and conversational passes; numerical evidence remains computed locally.")
    elif llm_provider.startswith("Ollama"):
        st.caption("Local Ollama is the privacy-first no-key option. On Streamlit Community Cloud, 127.0.0.1 refers to the cloud server, not your laptop; use a reachable/self-hosted Ollama endpoint or run the app locally.")
    llm_max_tokens = st.number_input("AI maximum output tokens", min_value=256, max_value=12000, value=4000, step=256, key="global_llm_max_tokens")
    st.session_state["llm_config"] = {
        "provider": llm_provider, "api_key": llm_api_key if needs_key else "", "model": llm_model.strip(),
        "base_url": llm_base_url.strip(), "max_tokens": int(llm_max_tokens),
    }
    if llm_configured(st.session_state["llm_config"]):
        st.success("AI engine configured for this session. Agentic Mode can now use it for evidence-grounded synthesis, conversation, RQ generation and paper refinement.")
        if st.button("TEST AI CONNECTION", key=f"global_llm_test_{llm_provider}"):
            try:
                probe = llm_reply(
                    "Return exactly CONNECTION_OK and no other text.",
                    st.session_state["llm_config"],
                    system="This is a connectivity test. Return exactly CONNECTION_OK.",
                    timeout=30,
                )
                if "CONNECTION_OK" in str(probe):
                    st.success(f"Connection successful: {llm_provider} · {llm_model.strip()}")
                else:
                    st.warning("The provider responded, but not with the expected connectivity token. The API route is reachable; inspect the model response/settings if generation behaves unexpectedly.")
            except Exception as exc:
                st.error(f"AI connection test failed: {exc}")
    else:
        st.caption("The whole application still works without AI. Configure a hosted key or a reachable Ollama model only when you want AI-level synthesis.")

    st.divider()
    pages = [
        "1. Executive overview", "2. Data hub & audit", "3. Research questions",
        "4. Descriptive statistics", "5. Hypothesis tests", "6. OLS & econometric laboratory",
        "6A. Monte Carlo & uncertainty",
        "7. 1,000 × 1,000 batch engine", "8. Original R&D regional panel",
        "8A. Panel model laboratory",
        "9. Detailed Greece GIS", "10. Time series & multivariate",
        "10A. Advanced clustering & segmentation", "10B. Predictive model laboratory",
        "11. Publication figures & HTML report", "12. Scenario & allocation engine",
        "12A. Dedicated MCDA engine",
        "12A.1 ITA / public-funding decision support",
        "12A.1B GAMS-compatible ITA Studio",
        "12A.2 Expert respondent analytics",
        "12B. Research Command Chair",
        "12C. Frontier methods laboratory",
        "12D. Agentic Research Mode",
        "13. Methods & reproducibility",
    ]
    page = st.radio("Module", pages)
    with st.expander("Quick operating guide", expanded=False):
        st.markdown("1. Upload or select data.\n2. Audit variables.\n3. Choose a research question.\n4. Run the matching method.\n5. Read assumptions and diagnostics.\n6. Download tables, figures and the reproducibility record.")
        st.caption("Interactive figures also expose Plotly's camera icon in the top-right toolbar. Use the explicit publication bundle for 600-dpi and vector output.")
    if not df.empty:
        st.caption(f"Active: {selected_label}\n\n{len(df):,} rows × {df.shape[1]:,} variables")


if df.empty and page not in {"12B. Research Command Chair", "12D. Agentic Research Mode"}:
    st.info("Upload one or more Excel/CSV files in the sidebar. The app accepts multiple files and multiple workbook sheets simultaneously.")
    st.subheader("Research programme already encoded")
    guided_dataframe("Encoded research programme", RQS)
    st.stop()

numeric = list(df.select_dtypes(include=np.number).columns)
categorical = [c for c in df.columns if c not in numeric]


if page == "1. Executive overview":
    module_guide(
        "A navigational summary of the active dataset and the full analytical workbench.",
        "Confirm the record and variable counts, inspect data readiness, then use the method navigator below to choose an analysis.",
        "This page does not estimate causal effects. It identifies data issues and directs you to the appropriate module.",
    )
    q = quality_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Variables", f"{df.shape[1]:,}")
    c3.metric("Numeric", f"{len(numeric):,}")
    c4.metric("Categorical", f"{len(categorical):,}")
    c5.metric("Encoded source RQs", len(RQS))
    st.subheader("Analysis navigator")
    navigator = pd.DataFrame([
        ["OLS & robust regression", "6. OLS & econometric laboratory", "Continuous outcomes; transparent conditional associations", "Coefficient, CI, diagnostics, fitted/residual evidence"],
        ["Monte Carlo OLS", "6A. Monte Carlo & uncertainty", "Sampling uncertainty, stability and heteroskedasticity-sensitive simulation", "Full draws, bias, MC SE, percentile intervals"],
        ["Monte Carlo portfolio", "6A. Monte Carlo & uncertainty", "R&D project selection under uncertain cost and benefit", "Selection probabilities and portfolio-risk distribution"],
        ["Logit / Probit / Counts", "6. OLS & econometric laboratory", "Binary, fractional or count outcomes", "GLM coefficients, fit and dispersion diagnostics"],
        ["1,000 × 1,000 OLS", "7. 1,000 × 1,000 batch engine", "High-dimensional discovery and screening", "All coefficient cells plus multiplicity correction"],
        ["Regional panel", "8. Original R&D regional panel", "EE1–EE9 region-year questions", "Panel tables, original model presets and trends"],
        ["FE / RE panel models", "8A. Panel model laboratory", "Repeated entity-time observations", "Pooled, two-way FE, RE and Hausman evidence"],
        ["Greece spatial analysis", "9. Detailed Greece GIS", "NUTS-2/NUTS-3 geographical patterns", "Maps, Moran/LISA tables and publication files"],
        ["Time series / PCA / legacy clusters", "10. Time series & multivariate", "Temporal and multivariate structure", "Exact test, loading, forecast and cluster tables"],
        ["Advanced clustering", "10A. Advanced clustering & segmentation", "One or many absorption/finance/innovation variables", "Automatic k, four algorithms, profiles and stability diagnostics"],
        ["Predictive model comparison", "10B. Predictive model laboratory", "Out-of-sample outcome prediction", "Seven models, CV metrics and permutation importance"],
        ["Multi-criteria decision analysis", "12A. Dedicated MCDA engine", "Transparent ranking of projects, regions or policy alternatives", "MAVT/TOPSIS/PROMETHEE-II, AHP/objective weights and robustness evidence"],
        ["Frontier causal / Bayesian / Pareto / XAI", "12C. Frontier methods laboratory", "Questions requiring doubly robust causal estimation, posterior uncertainty, multi-objective frontiers or explainable ML", "AIPW, posterior draws/predictive checks, Pareto portfolios, SHAP and large-data outputs"],
        ["Agentic end-to-end research", "12D. Agentic Research Mode", "Fast evidence-to-submission workflow with dataset + literature PDFs", "Up to 150 RQs, approved analysis plan, executed models, discussion/conclusions, DOCX/XLSX/JSON/HTML/graphics bundle"],
    ], columns=["method", "module", "use_when", "principal_outputs"])
    guided_dataframe("Analysis navigator", navigator)
    st.download_button("Download method navigator", to_excel_bytes({"Method navigator": navigator}), "method_navigator.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Data readiness")
    guided_dataframe("Data readiness", q)
    left, right = st.columns(2)
    with left:
        st.subheader("Missingness: leading variables")
        miss = data_dictionary(df).sort_values("missing_pct", ascending=False).head(20)
        missing_fig = px.bar(miss.sort_values("missing_pct"), x="missing_pct", y="variable", orientation="h", title="Missing observations (%)")
        figure_with_downloads(missing_fig, "executive_missingness", miss, "High missingness can change both the estimand and the effective sample. Review whether missing values are structural, random or coding artefacts.")
    with right:
        st.subheader("Dataset composition")
        comp = pd.DataFrame({"type": ["Numeric", "Categorical/text"], "variables": [len(numeric), len(categorical)]})
        composition_fig = px.bar(comp, x="type", y="variables", color="type", title="Variable types")
        figure_with_downloads(composition_fig, "executive_variable_composition", comp, "Numeric variables are immediately available to quantitative models; categorical variables can enter as controls or fixed effects where supported.")
    if is_rd_dataset(df):
        st.success("The original 83-variable Makryvelios R&D structure was detected. The dedicated regional-panel and research-question presets are enabled.")
    st.caption("Default view is descriptive. No result is interpreted causally unless an explicit identification strategy is supplied.")


elif page == "2. Data hub & audit":
    module_guide("Verify file structure, variable types, missingness and duplicates before analysis.", "Review the file inventory, quality checks, dictionary and record preview; export the cleaned analytical view when satisfied.", "Incorrect data types or joins invalidate downstream results. Resolve them here before modelling.")
    st.subheader("Loaded files and sheets")
    inventory = pd.DataFrame([{"dataset": name, "rows": len(frame), "variables": frame.shape[1]} for name, frame in frames.items()])
    guided_dataframe("Loaded files and sheets", inventory)
    st.subheader("Quality checks")
    guided_dataframe("Quality checks", quality_summary(df))
    dictionary = data_dictionary(df)
    st.subheader("Variable dictionary")
    guided_dataframe("Variable dictionary", dictionary)
    st.subheader("Record preview")
    guided_dataframe("Record preview", df.head(200), max_rows=200)
    st.subheader("Robust outlier surveillance")
    outliers = outlier_summary(df, numeric)
    st.caption("IQR fences flag unusual univariate values without deleting or winsorising them. Genuine large infrastructure projects must not be treated as data errors automatically.")
    guided_dataframe("Robust outlier surveillance", outliers, max_rows=500)
    if not outliers.empty:
        outlier_fig = px.bar(outliers.head(40).sort_values("outlier_pct"), x="outlier_pct", y="variable", orientation="h", title="Variables with the highest IQR outlier shares")
        figure_with_downloads(outlier_fig, "data_audit_outliers", outliers, "Outlier flags are screening indicators. Inspect raw records and domain plausibility before transformation or exclusion.")
    a, b, c, d = st.columns(4)
    with a: download_table("Download active data (CSV)", df, "active_dataset.csv")
    with b: download_table("Download dictionary", dictionary, "variable_dictionary.csv")
    with c: st.download_button("Download machine summary", serialisable_summary(df), "dataset_summary.json", "application/json")
    with d: st.download_button("Download complete audit workbook", to_excel_bytes({"Quality": quality_summary(df), "Dictionary": dictionary, "Outlier surveillance": outliers, "Preview": df.head(10_000)}), "complete_data_audit.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


elif page == "3. Research questions":
    module_guide("Connect each analysis to the documented R&D and Antonis Tritsis research programme.", "Filter the programme, inspect the exact question and hypotheses, then export the catalogue for the methods or manuscript file.", "Questions and hypotheses should be specified before inspecting model significance to reduce data-driven inference.")
    st.subheader("Recovered R&D questions and extended Antonis Tritsis programme")
    programme = st.multiselect("Programme", sorted(RQS.programme.unique()), default=sorted(RQS.programme.unique()))
    questions = RQS[RQS.programme.isin(programme)]
    guided_dataframe("Research questions", questions)
    st.subheader("Source hypotheses: R&D EE1–EE9")
    guided_dataframe("Source hypotheses", HYPOTHESES)
    st.caption("EE1–EE9 and their hypotheses are transcribed from the supplied Stata research-question document. AT-RQ1–AT-RQ11 are explicitly labelled reconstructions from the documented 1,454-project dataset scope; AT-RQ12 is a new cross-dataset question.")
    st.download_button("Download question catalogue", to_excel_bytes({"Research questions": RQS, "Hypotheses": HYPOTHESES}), "research_questions_and_hypotheses.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


elif page == "4. Descriptive statistics":
    module_guide("Summarise distributions, categorical composition and bivariate associations.", "Select numeric and categorical variables, inspect the distribution, then choose the correlation measure appropriate to scale and monotonicity.", "Correlation is pairwise association, not an adjusted or causal effect. Outliers and repeated regional values can dominate it.")
    st.subheader("Numeric summaries")
    chosen = st.multiselect("Numeric variables", numeric, default=numeric[:min(12, len(numeric))], max_selections=1000)
    desc = descriptive_statistics(df, chosen)
    guided_dataframe("Descriptive statistics", desc)
    if chosen:
        variable = st.selectbox("Distribution variable", chosen)
        fig = px.histogram(df, x=variable, marginal="box", nbins=40, title=f"Distribution: {variable}", color_discrete_sequence=["#155B8A"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,43,60,.78)")
        figure_with_downloads(fig, "descriptive_distribution", df[[variable]], "Use the histogram, box summary and extreme observations together. Skewed financial variables often require logarithmic transformation for regression.")
    st.subheader("Categorical frequencies")
    cats = st.multiselect("Categorical variables", categorical, default=categorical[:min(3, len(categorical))])
    cat_table = categorical_summary(df, cats)
    guided_dataframe("Categorical frequencies", cat_table)
    if chosen:
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        corr_vars = chosen[:100]
        corr, pvals = correlation_matrix(df, corr_vars, method)
        st.subheader(f"{method.title()} correlation matrix")
        guided_dataframe(f"{method.title()} correlation matrix", corr.reset_index())
        corr_fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", title=f"{method.title()} correlations")
        figure_with_downloads(corr_fig, "descriptive_correlation_heatmap", corr.reset_index(), "Large absolute correlations may signal redundancy or multicollinearity, but they do not account for other variables or causal direction.")
        st.download_button("Download descriptive workbook", to_excel_bytes({"Descriptive": desc, "Frequencies": cat_table, "Correlations": corr.reset_index(), "Correlation p-values": pvals.reset_index()}), "descriptive_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


elif page == "5. Hypothesis tests":
    module_guide("Run classical and non-parametric comparisons, categorical association and normality diagnostics.", "Select an outcome and grouping variable, run the relevant tab, then export the resulting test table.", "Report effect size and assumptions alongside p-values; a small p-value is not evidence that an effect is substantively large.")
    tab1, tab2, tab3 = st.tabs(["Numeric outcomes by group", "Categorical association", "Normality"])
    with tab1:
        outcomes = st.multiselect("Outcomes", numeric, default=numeric[:min(5, len(numeric))], key="test_outcomes")
        group = st.selectbox("Grouping variable", categorical, key="test_group") if categorical else None
        if st.button("Run t/ANOVA and non-parametric tests") and group:
            result = group_tests(df, outcomes, group)
            st.session_state["group_tests"] = result
        result = st.session_state.get("group_tests", pd.DataFrame())
        guided_dataframe("Group-comparison tests", result)
    with tab2:
        group2 = st.selectbox("Reference categorical variable", categorical, key="chi_group") if categorical else None
        vars2 = st.multiselect("Variables to cross-tabulate", [c for c in categorical if c != group2], key="chi_vars")
        if st.button("Run chi-square tests") and group2:
            st.session_state["chi_tests"] = chi_square_tests(df, vars2, group2)
        guided_dataframe("Categorical association tests", st.session_state.get("chi_tests", pd.DataFrame()))
    with tab3:
        nv = st.multiselect("Variables", numeric, default=numeric[:min(5, len(numeric))], key="normal_vars")
        if st.button("Run normality tests"):
            st.session_state["normal_tests"] = normality_tests(df, nv)
        guided_dataframe("Normality tests", st.session_state.get("normal_tests", pd.DataFrame()))
    st.info("With large samples, normality tests detect trivial deviations. Inspect distributions and use robust/appropriate outcome models rather than treating p-values mechanically.")
    test_tables = {
        "Group tests": st.session_state.get("group_tests", pd.DataFrame()),
        "Chi-square tests": st.session_state.get("chi_tests", pd.DataFrame()),
        "Normality tests": st.session_state.get("normal_tests", pd.DataFrame()),
    }
    if any(not t.empty for t in test_tables.values()):
        st.download_button("Download all hypothesis-test tables", to_excel_bytes({k: v for k, v in test_tables.items() if not v.empty}), "hypothesis_test_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    group_effects = test_tables["Group tests"]
    if not group_effects.empty and "effect_size" in group_effects:
        effect_plot = group_effects.dropna(subset=["effect_size"]).copy()
        if not effect_plot.empty:
            effect_fig = px.bar(effect_plot, x="effect_size", y="outcome", color="test", orientation="h", hover_data=["effect_metric", "p_value"], title="Group-comparison effect sizes")
            figure_with_downloads(effect_fig, "hypothesis_effect_sizes", effect_plot, "Effect sizes quantify magnitude. Their scales differ by test, so compare values within the same effect metric rather than mechanically across metrics.")
    chi_effects = test_tables["Chi-square tests"]
    if not chi_effects.empty:
        chi_fig = px.bar(chi_effects.sort_values("cramers_v"), x="cramers_v", y="variable", orientation="h", color="p_value", title="Categorical association strength (Cramér's V)", color_continuous_scale="Tealgrn_r")
        figure_with_downloads(chi_fig, "categorical_association_effects", chi_effects, "Cramér's V ranges from 0 to 1. Interpret magnitude in context and inspect sparse expected counts before relying on chi-square inference.")


elif page == "6. OLS & econometric laboratory":
    st.subheader("OLS Studio and advanced econometric laboratory")
    module_guide(
        "Estimate ordinary least squares visibly and directly, or select an outcome-appropriate alternative estimator.",
        "Choose the dependent variable, estimator, predictors and covariance rule. Start with OLS + HC3 for a continuous outcome, then press Estimate model.",
        "A coefficient is a conditional association holding the included regressors constant. Statistical significance, fit and causality are different questions; read the diagnostic and explanatory panels together.",
    )
    st.markdown("**OLS is available here as the default estimator.** HC3 is the recommended general-purpose robust covariance for cross-sectional project data; clustered covariance is preferable when observations share a region, organisation or time unit.")
    c1, c2 = st.columns([1, 2])
    with c1:
        y = st.selectbox("Dependent variable", numeric)
        estimator = st.selectbox("Estimator", ["OLS", "WLS", "Robust Huber", "Logit", "Probit", "Poisson", "Negative binomial", "Fractional logit", "Gamma log-link", "Quantile regression"])
        covariance = st.selectbox("Covariance", ["HC3", "HC2", "HC1", "HC0", "HAC", "nonrobust"])
        quantile = st.slider("Quantile", .05, .95, .50, .05) if estimator == "Quantile regression" else .5
    with c2:
        x = st.multiselect("Independent variables (up to 1,000)", [c for c in numeric if c != y], default=[c for c in numeric if c != y][:min(7, max(len(numeric)-1, 0))], max_selections=1000)
        cats = st.multiselect("Categorical controls / fixed effects", categorical, help="Select region and year here to estimate least-squares dummy-variable fixed effects.")
        cluster = st.selectbox("Cluster standard errors by", [None] + list(df.columns))
        weights = st.selectbox("WLS weights", [None] + numeric) if estimator == "WLS" else None
    estimator_notes = {
        "OLS": "Continuous outcome. Coefficients are changes in the expected outcome per one-unit change in a regressor.",
        "WLS": "Continuous outcome with known or defensible observation weights.",
        "Robust Huber": "Continuous outcome estimated by Huber M-estimation to reduce the leverage of large residuals; this changes the estimator, not merely its standard errors.",
        "Logit": "Binary outcome coded 0/1; coefficients are in log-odds units.",
        "Probit": "Binary outcome coded 0/1 under a normal latent-index link.",
        "Poisson": "Non-negative count outcome; exponentiated coefficients are incidence-rate ratios.",
        "Negative binomial": "Overdispersed non-negative counts where variance materially exceeds the mean.",
        "Fractional logit": "Proportion or rate bounded between 0 and 1, including endpoints.",
        "Gamma log-link": "Strictly positive, right-skewed continuous outcome; exponentiated coefficients are multiplicative mean ratios.",
        "Quantile regression": "Conditional quantiles; useful when effects differ across the outcome distribution.",
    }
    st.caption(estimator_notes[estimator])
    if len(x) > 250:
        st.warning("A detailed 250+ regressor model may be slow or rank-deficient. The separate batch engine is optimised for very wide specifications.")
    if st.button("Estimate model", type="primary", disabled=not x):
        try:
            with st.spinner("Estimating and running diagnostics…"):
                out = fit_detailed_model(df, y, x[:MAX_INDEPENDENT], cats, estimator, covariance, cluster, weights, quantile)
            st.session_state["model_output"] = out
            st.session_state["model_signature"] = {"y": y, "x": x, "categorical": cats, "estimator": estimator}
        except Exception as exc:
            st.error(f"Model failed: {exc}")
    out = st.session_state.get("model_output")
    if out:
        table_with_downloads("Model fit", out.fit, "econometric_model_fit", "Fit statistics describe the estimated sample specification; they do not establish causal validity.")
        table_with_downloads("Coefficient table", out.coefficients, "econometric_coefficients", "The confidence interval and robust p-value should be read together with effect size, units and multiplicity.")
        table_with_downloads("Model diagnostics", out.diagnostics, "econometric_diagnostics", "Diagnostic tests flag possible violations; they do not automatically prescribe a single correction.")
        st.subheader("Explanatory comments")
        for comment in out.interpretation: st.info(comment)
        fit_fig = px.scatter(out.predictions, x="fitted", y="observed", trendline="ols", title="Observed versus fitted", color_discrete_sequence=["#087f95"])
        figure_with_downloads(fit_fig, "observed_vs_fitted", out.predictions, "Points close to the fitted trend indicate better in-sample agreement; systematic curvature or widening dispersion suggests misspecification or heteroskedasticity.")
        residual_fig = px.scatter(out.predictions, x="fitted", y="residual", title="Residuals versus fitted values", color_discrete_sequence=["#d89b2b"])
        residual_fig.add_hline(y=0, line_dash="dash", line_color="#263746")
        figure_with_downloads(residual_fig, "residuals_vs_fitted", out.predictions, "A desirable residual plot is centred around zero without systematic curvature or a funnel pattern.")
        coef_plot = out.coefficients[out.coefficients.term != "const"].copy()
        if not coef_plot.empty:
            forest = px.scatter(coef_plot, x="coefficient", y="term", error_x=coef_plot.ci_95_high - coef_plot.coefficient, error_x_minus=coef_plot.coefficient - coef_plot.ci_95_low, title="Coefficients with 95% confidence intervals", color_discrete_sequence=["#087f95"])
            forest.add_vline(x=0, line_dash="dash", line_color="#263746")
            figure_with_downloads(forest, "coefficient_forest", coef_plot, "Intervals crossing zero are not statistically distinguishable from zero at the corresponding two-sided 5% level under the selected covariance estimator.")
        with st.expander("Multicollinearity: VIF"):
            signature = st.session_state.get("model_signature", {})
            guided_dataframe("Variance inflation factors", vif_table(df, signature.get("x", []), signature.get("categorical", [])))
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download complete model workbook", to_excel_bytes({"Fit": out.fit, "Coefficients": out.coefficients, "Diagnostics": out.diagnostics, "Predictions": out.predictions}), "econometric_model_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d2:
            if st.session_state.get("model_signature", {}).get("estimator") == "OLS":
                st.download_button("Download OLS publication bundle (colour + B&W)", ols_publication_bundle(out.predictions, out.coefficients, f"OLS model: {st.session_state.get('model_signature', {}).get('y', '')}"), "ols_publication_bundle.zip", "application/zip")
    st.divider()
    st.subheader("Advanced identification and regularisation")
    iv_tab, did_tab, reg_tab = st.tabs(["IV / 2SLS", "Difference-in-differences", "Ridge / Lasso / Elastic Net"])
    with iv_tab:
        iv_y = st.selectbox("IV outcome", numeric, key="iv_y")
        iv_endog = st.selectbox("Endogenous regressor", [c for c in numeric if c != iv_y], key="iv_endog")
        iv_inst = st.multiselect("Excluded instrument(s)", [c for c in numeric if c not in {iv_y, iv_endog}], key="iv_inst")
        iv_exog = st.multiselect("Exogenous controls", [c for c in numeric if c not in {iv_y, iv_endog} and c not in iv_inst], key="iv_exog")
        if st.button("Estimate 2SLS", disabled=not iv_inst):
            try: st.session_state["iv_result"] = instrumental_variables_2sls(df, iv_y, iv_endog, iv_inst, iv_exog)
            except Exception as exc: st.error(str(exc))
        if "iv_result" in st.session_state:
            iv_coef, iv_fit = st.session_state["iv_result"]
            guided_dataframe("Instrumental-variables fit", iv_fit)
            guided_dataframe("Instrumental-variables coefficients", iv_coef)
            st.warning("Instrument relevance is necessary but not sufficient: exclusion and independence cannot be established by the first-stage F statistic alone.")
            st.download_button("Download IV/2SLS workbook", to_excel_bytes({"First and second stage fit": iv_fit, "IV coefficients": iv_coef}), "iv_2sls_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with did_tab:
        did_y = st.selectbox("DiD outcome", numeric, key="did_y")
        did_treat = st.selectbox("Treatment indicator", [c for c in numeric if c != did_y], key="did_treat")
        did_post = st.selectbox("Post-period indicator", [c for c in numeric if c not in {did_y, did_treat}], key="did_post")
        did_controls = st.multiselect("DiD controls", [c for c in numeric if c not in {did_y, did_treat, did_post}], key="did_controls")
        did_cluster = st.selectbox("DiD cluster", [None] + list(df.columns), key="did_cluster")
        if st.button("Estimate DiD"):
            try: st.session_state["did_result"] = difference_in_differences(df, did_y, did_treat, did_post, did_controls, did_cluster)
            except Exception as exc: st.error(str(exc))
        if "did_result" in st.session_state:
            did = st.session_state["did_result"]
            guided_dataframe("Difference-in-differences coefficients", did.coefficients)
            for comment in did.interpretation: st.info(comment)
            st.download_button("Download difference-in-differences workbook", to_excel_bytes({"Fit": did.fit, "Coefficients": did.coefficients, "Diagnostics": did.diagnostics, "Predictions": did.predictions}), "difference_in_differences_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with reg_tab:
        reg_y = st.selectbox("Regularised outcome", numeric, key="reg_y")
        reg_x = st.multiselect("Predictors", [c for c in numeric if c != reg_y], default=[c for c in numeric if c != reg_y][:min(20, len(numeric))], max_selections=1000, key="reg_x")
        reg_method = st.selectbox("Penalty", ["Ridge", "Lasso", "Elastic Net"], key="reg_method")
        reg_alpha = st.number_input("Penalty strength (alpha)", min_value=.0001, value=1.0, format="%.4f")
        reg_l1 = st.slider("L1 ratio", 0.0, 1.0, .5) if reg_method == "Elastic Net" else .5
        if st.button("Fit regularised model", disabled=not reg_x):
            try: st.session_state["regularised"] = regularised_regression(df, reg_y, reg_x, reg_method, reg_alpha, reg_l1)
            except Exception as exc: st.error(str(exc))
        if "regularised" in st.session_state:
            reg_coef, reg_fit = st.session_state["regularised"]
            guided_dataframe("Regularised model fit", reg_fit)
            guided_dataframe("Regularised coefficients", reg_coef)
            st.info("Regularised coefficients are standardised predictive shrinkage estimates. Their magnitude depends on alpha and, for Elastic Net, the L1 ratio; they are not conventional unbiased structural coefficients.")
            st.download_button("Download regularised-model workbook", to_excel_bytes({"Performance": reg_fit, "Standardised coefficients": reg_coef}), "regularised_model_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


elif page == "6A. Monte Carlo & uncertainty":
    st.subheader("Monte Carlo simulation and robustness laboratory")
    module_guide(
        "Quantify how regression estimates or R&D portfolio decisions vary when sampling and input uncertainty are propagated repeatedly.",
        "Choose either OLS uncertainty or portfolio selection, set a reproducible seed and the number of simulations, then run the engine. Begin with 1,000–2,000 replications and increase only after the specification is stable.",
        "Monte Carlo describes uncertainty conditional on the assumed data-generating process. It does not repair omitted variables, invalid measurements or an indefensible causal design.",
    )
    mc_reg, mc_port = st.tabs(["OLS coefficient uncertainty", "R&D portfolio selection"])
    with mc_reg:
        st.markdown("#### Simulation-based OLS stability")
        st.caption("Wild bootstrap is the default because project-level economic data commonly exhibit heteroskedasticity. Residual bootstrap assumes exchangeable residuals; parametric normal simulation imposes a normal error distribution.")
        mc_y = st.selectbox("Dependent variable", numeric, key="mc_y")
        mc_candidates = [c for c in numeric if c != mc_y]
        mc_x = st.multiselect("Independent variables (maximum 100)", mc_candidates, default=mc_candidates[:min(2, len(mc_candidates))], max_selections=100, key="mc_x")
        a, b, c, d = st.columns(4)
        with a: mc_method = st.selectbox("Simulation method", ["Wild bootstrap", "Residual bootstrap", "Parametric normal"], key="mc_method")
        with b: mc_sims = st.slider("Replications", 100, 10_000, 2_000, 100, key="mc_sims")
        with c: mc_conf = st.selectbox("Confidence level", [.90, .95, .99], index=1, format_func=lambda v: f"{v:.0%}", key="mc_conf")
        with d: mc_seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="mc_seed")
        if st.button("Run Monte Carlo OLS", type="primary", disabled=not mc_x):
            try:
                with st.spinner(f"Running {mc_sims:,} reproducible simulations…"):
                    st.session_state["mc_ols"] = monte_carlo_ols(df, mc_y, mc_x, mc_sims, mc_method, mc_conf, int(mc_seed))
                    st.session_state["mc_ols_signature"] = {"y": mc_y, "x": mc_x, "method": mc_method}
            except Exception as exc:
                st.error(f"Monte Carlo OLS failed: {exc}")
        if "mc_ols" in st.session_state:
            mc_summary, mc_draws, mc_fit = st.session_state["mc_ols"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Simulations", f"{int(mc_fit.simulations.iloc[0]):,}")
            m2.metric("Complete observations", f"{int(mc_fit.complete_observations.iloc[0]):,}")
            m3.metric("Predictors", f"{int(mc_fit.predictors.iloc[0]):,}")
            m4.metric("Base OLS R²", f"{float(mc_fit.base_ols_r_squared.iloc[0]):.3f}")
            table_with_downloads("Monte Carlo model settings", mc_fit, "monte_carlo_ols_settings", "The seed and method make the simulation exactly reproducible.")
            table_with_downloads("Coefficient uncertainty summary", mc_summary, "monte_carlo_ols_summary", "Simulation bias, empirical standard error, percentile interval and sign probability are reported for every coefficient.")
            selected_term = st.selectbox("Coefficient to visualise", list(mc_draws.columns[1:]), key="mc_term")
            hist = px.histogram(mc_draws, x=selected_term, nbins=60, marginal="box", histnorm="probability density", title=f"Monte Carlo distribution: {selected_term}", color_discrete_sequence=["#087f95"])
            hist.add_vline(x=0, line_dash="dot", line_color="#263746")
            figure_with_downloads(hist, "monte_carlo_coefficient_distribution", mc_draws[["simulation", selected_term]], "The width represents simulated sampling uncertainty. A distribution concentrated on one side of zero indicates greater sign stability under the selected simulation mechanism.")
            row = mc_summary.loc[mc_summary.term == selected_term].iloc[0]
            st.info(f"For {selected_term}, the simulated mean is {row.simulation_mean:.5g}, the Monte Carlo standard error is {row.monte_carlo_se:.5g}, and the probability of a positive coefficient is {row.probability_positive:.1%}.")
            e1, e2 = st.columns(2)
            with e1:
                st.download_button("Download complete Monte Carlo workbook", to_excel_bytes({"Settings": mc_fit, "Coefficient summary": mc_summary, "All simulation draws": mc_draws}), "monte_carlo_ols_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with e2:
                st.download_button("Download Monte Carlo publication bundle", monte_carlo_publication_bundle(mc_draws, mc_summary, selected_term), "monte_carlo_publication_bundle.zip", "application/zip")
    with mc_port:
        st.markdown("#### Stochastic R&D portfolio selection")
        st.caption("Each replication perturbs cost and benefit, ranks projects by simulated benefit-to-cost ratio and fills the available budget. The selection probability measures robustness across uncertain scenarios.")
        p1, p2, p3 = st.columns(3)
        with p1: port_cost = st.selectbox("Project cost variable", numeric, key="port_cost")
        with p2: port_benefit = st.selectbox("Project benefit / score variable", [c for c in numeric if c != port_cost], key="port_benefit")
        id_options = [None] + list(df.columns)
        with p3: port_id = st.selectbox("Project identifier (optional)", id_options, key="port_id")
        positive_costs = pd.to_numeric(df[port_cost], errors="coerce").loc[lambda s: s > 0]
        default_budget = float(positive_costs.sum() * .25) if len(positive_costs) else 1_000_000.0
        q1, q2, q3, q4 = st.columns(4)
        with q1: port_budget = st.number_input("Available budget", min_value=0.01, value=max(default_budget, .01), format="%.2f", key="port_budget")
        with q2: port_sims = st.slider("Portfolio replications", 100, 5_000, 1_000, 100, key="port_sims")
        with q3: port_cost_cv = st.slider("Cost uncertainty (CV)", 0.0, 1.0, .10, .01, key="port_cost_cv")
        with q4: port_benefit_cv = st.slider("Benefit uncertainty (CV)", 0.0, 1.0, .20, .01, key="port_benefit_cv")
        correlation = st.slider("Correlation between cost and benefit shocks", -.90, .90, 0.0, .05, key="port_corr")
        port_seed = st.number_input("Portfolio random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="port_seed")
        if st.button("Run stochastic portfolio", type="primary"):
            try:
                with st.spinner(f"Evaluating {port_sims:,} uncertain portfolios…"):
                    st.session_state["mc_portfolio"] = monte_carlo_portfolio(df, port_cost, port_benefit, port_budget, port_id, port_sims, port_cost_cv, port_benefit_cv, correlation, int(port_seed))
            except Exception as exc:
                st.error(f"Portfolio simulation failed: {exc}")
        if "mc_portfolio" in st.session_state:
            port_summary, projects, simulations = st.session_state["mc_portfolio"]
            r = port_summary.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Eligible projects", f"{int(r.eligible_projects):,}")
            k2.metric("Mean portfolio benefit", f"{r.mean_portfolio_benefit:,.2f}")
            k3.metric("5th percentile benefit", f"{r.p05_portfolio_benefit:,.2f}")
            k4.metric("Mean projects selected", f"{r.mean_projects_selected:,.1f}")
            table_with_downloads("Portfolio simulation summary", port_summary, "monte_carlo_portfolio_summary", "The 5th percentile is a downside-oriented benefit scenario; it is not a guaranteed minimum.")
            table_with_downloads("Project selection robustness", projects, "monte_carlo_project_selection", "High selection probability indicates that a project remains attractive across many cost-benefit perturbations.")
            top = projects.head(40).sort_values("selection_probability")
            selection_fig = px.bar(top, x="selection_probability", y="project_id", orientation="h", title="Most robust project selections", color="selection_probability", color_continuous_scale=["#cbeaf0", "#087f95"])
            figure_with_downloads(selection_fig, "portfolio_selection_probability", top, "Selection probability is conditional on the budget, uncertainty parameters and ratio-based decision rule selected above.")
            benefit_fig = px.histogram(simulations, x="portfolio_benefit", nbins=50, marginal="box", title="Distribution of total portfolio benefit", color_discrete_sequence=["#d89b2b"])
            figure_with_downloads(benefit_fig, "portfolio_benefit_distribution", simulations, "The distribution displays the outcome risk generated by cost and benefit uncertainty across all replications.")
            st.download_button("Download complete portfolio workbook", to_excel_bytes({"Summary": port_summary, "Project probabilities": projects, "Simulation portfolios": simulations}), "monte_carlo_portfolio_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.warning("This implementation deliberately exposes its assumptions. A policy-ready portfolio should add eligibility, minimum/maximum allocations, thematic balance, regional equity and other programme constraints before decisions are taken.")


elif page == "7. 1,000 × 1,000 batch engine":
    module_guide("Screen up to 1,000 outcomes against up to 1,000 predictors using one vectorised OLS design.", "Choose outcomes and predictors, select missing-data and multiplicity rules, then run and download the complete workbook.", "This is discovery screening. Re-estimate shortlisted relationships in the OLS Studio with robust/clustered inference and theory-led controls.")
    st.subheader("High-dimensional multi-outcome OLS screening")
    st.write("Select up to 1,000 dependent variables and 1,000 independent variables simultaneously. The engine performs one SVD/pseudoinverse and vectorised inference rather than fitting one million separate models.")
    ys = st.multiselect("Dependent variables", numeric, default=numeric[:min(3, len(numeric))], max_selections=MAX_DEPENDENT, key="batch_y")
    xs = st.multiselect("Independent variables", [c for c in numeric if c not in ys], default=[c for c in numeric if c not in ys][:min(10, len(numeric))], max_selections=MAX_INDEPENDENT, key="batch_x")
    missing = st.selectbox("Missing-data rule", ["median imputation", "complete cases"])
    adjustment = st.selectbox("Multiple-testing adjustment", ["Benjamini–Hochberg", "Bonferroni"])
    workload = len(ys) * (len(xs) + 1)
    st.metric("Coefficient cells", f"{workload:,}")
    if st.button("Run batch engine", type="primary", disabled=not ys or not xs):
        try:
            with st.spinner("Running vectorised multi-outcome model…"):
                coef, fit = matrix_ols_many_outcomes(df, ys, xs, missing)
                coef["p_adjusted"] = p_adjust(coef.p_value, adjustment)
                coef["significant_adjusted_5pct"] = coef.p_adjusted < .05
            st.session_state["batch_coef"] = coef; st.session_state["batch_fit"] = fit
        except Exception as exc:
            st.error(str(exc))
    coef = st.session_state.get("batch_coef", pd.DataFrame()); fit = st.session_state.get("batch_fit", pd.DataFrame())
    if not coef.empty:
        guided_dataframe("High-dimensional model fit", fit)
        show = coef.sort_values("p_adjusted").head(10_000)
        guided_dataframe("High-dimensional coefficient screen", show)
        st.caption("The on-screen table is capped at 10,000 rows; the download contains all coefficient cells.")
        st.download_button("Download complete batch workbook", to_excel_bytes({"Model fit": fit, "Coefficients": coef}), "batch_1000x1000_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.warning("This batch mode is a screening device using conventional homoskedastic OLS standard errors. Confirm shortlisted specifications in the detailed laboratory with robust/clustered inference and theory-led controls.")


elif page == "8. Original R&D regional panel":
    module_guide("Reconstruct the documented EE1–EE9 region-year analytical panel without losing the original project-level data.", "Choose the allocation rule, inspect the recovered model presets and regional trend, then export the complete panel.", "The allocation rule changes timing and should be reported. Regional aggregation reduces the effective sample size and may amplify Attica's leverage.")
    if not is_rd_dataset(df):
        st.warning("The active dataset does not expose the required original R&D project/year/region columns. Select or upload the 83-variable Makryvelios workbook.")
    else:
        allocation = st.selectbox("Panel allocation", ["End-year", "Start-year", "Active years – even allocation"])
        try:
            panel = build_region_year_panel(df, allocation)
            st.session_state["rd_panel"] = panel
            c1, c2, c3 = st.columns(3)
            c1.metric("Panel rows", f"{len(panel):,}"); c2.metric("Regions", panel.region.nunique()); c3.metric("Years", f"{int(panel.year.min())}–{int(panel.year.max())}")
            st.subheader("Recovered EE1–EE9 model specifications")
            guided_dataframe("Recovered EE1–EE9 model specifications", original_model_presets(df, panel))
            metrics = [c for c in ["regional_absorption_index", "project_count", "collaborative_projects", "patents", "spin_offs", "smes_benefited", "participating_firms", "total_budget_allocated", "total_public_expenditure_allocated"] if c in panel]
            metric = st.selectbox("Panel metric", metrics)
            rule = "mean" if "index" in metric or "share" in metric else "sum"
            trend = panel.groupby("year")[metric].agg(rule).reset_index()
            panel_trend_fig = px.line(trend, x="year", y=metric, markers=True, title=f"R&D region–year trend: {metric}")
            figure_with_downloads(panel_trend_fig, "rd_region_year_trend", trend, "The national trend aggregates regional panel cells according to the displayed rule; inspect regional heterogeneity before drawing national conclusions.")
            guided_dataframe("Region–year analytical panel", panel)
            st.download_button("Download region–year panel", to_excel_bytes({"Region-year panel": panel, "Model presets": original_model_presets(df, panel)}), "rd_region_year_panel.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.error(f"Panel construction failed: {exc}")


elif page == "8A. Panel model laboratory":
    st.subheader("Longitudinal econometrics: pooled, fixed and random effects")
    module_guide(
        "Compare pooled OLS, two-way fixed effects and random effects on a declared entity-time panel.",
        "Choose entity and time identifiers, outcome, predictors, cell aggregation and covariance. Use region and year for the Makryvelios regional design, then inspect Hausman and within/between fit.",
        "Fixed-effects coefficients use within-entity change; random effects require orthogonality between unobserved entity effects and regressors. Neither is automatically causal.",
    )
    p1, p2, p3 = st.columns(3)
    with p1: panel_entity = st.selectbox("Entity identifier", list(df.columns), key="panel_entity")
    with p2: panel_time = st.selectbox("Time identifier", [c for c in df.columns if c != panel_entity], key="panel_time")
    with p3: panel_y = st.selectbox("Panel outcome", numeric, key="panel_y")
    panel_x_candidates = [c for c in numeric if c not in {panel_y, panel_entity, panel_time}]
    panel_x = st.multiselect("Time-varying predictors", panel_x_candidates, default=panel_x_candidates[:min(4, len(panel_x_candidates))], max_selections=100, key="panel_x")
    p4, p5 = st.columns(2)
    with p4: panel_agg = st.selectbox("Aggregate duplicate entity-time rows", ["Mean", "Sum", "Median"], key="panel_agg")
    with p5: panel_cov = st.selectbox("Covariance estimator", ["Clustered by entity", "Robust", "Unadjusted"], key="panel_cov")
    st.caption("If project-level records repeat within a region-year, the chosen aggregation creates one analytical cell. The exported prepared panel records source_rows for auditability.")
    if st.button("Estimate panel model suite", type="primary", disabled=not panel_x):
        try:
            with st.spinner("Estimating pooled OLS, two-way fixed effects, random effects and Hausman diagnostics…"):
                st.session_state["panel_suite"] = panel_model_suite(df, panel_entity, panel_time, panel_y, panel_x, panel_agg, panel_cov)
        except Exception as exc:
            st.error(f"Panel estimation failed: {exc}")
    if "panel_suite" in st.session_state:
        panel_fit, panel_coef, hausman, prepared_panel, panel_comments = st.session_state["panel_suite"]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Panel cells", f"{len(prepared_panel):,}")
        a2.metric("Entities", f"{prepared_panel[panel_entity].nunique():,}")
        a3.metric("Periods", f"{prepared_panel[panel_time].nunique():,}")
        a4.metric("Estimated models", f"{panel_fit.model.nunique():,}")
        table_with_downloads("Panel model fit", panel_fit, "panel_model_fit", "Within R² is central for fixed effects; overall R² is not directly comparable across every estimator definition.")
        table_with_downloads("Panel coefficients", panel_coef, "panel_coefficients", "Coefficients and intervals use the selected covariance estimator. Compare signs, magnitudes and uncertainty across specifications.")
        if not hausman.empty:
            table_with_downloads("Hausman specification test", hausman, "panel_hausman", "A low p-value indicates systematic FE–RE coefficient differences under the test assumptions.")
        st.subheader("Panel interpretation")
        for comment in panel_comments: st.info(comment)
        panel_plot = panel_coef[panel_coef.term != "const"].copy()
        if not panel_plot.empty:
            fig = px.scatter(panel_plot, x="coefficient", y="term", color="model", error_x=panel_plot.ci_95_high - panel_plot.coefficient, error_x_minus=panel_plot.coefficient - panel_plot.ci_95_low, title="Panel coefficient comparison")
            fig.add_vline(x=0, line_dash="dash", line_color="#D7E5EA")
            figure_with_downloads(fig, "panel_coefficient_comparison", panel_plot, "Material disagreement between pooled, fixed and random-effects coefficients is substantively informative and should be explained, not hidden.")
        b1, b2 = st.columns(2)
        with b1:
            st.download_button("Download complete panel workbook", to_excel_bytes({"Model fit": panel_fit, "Coefficients": panel_coef, "Hausman": hausman, "Prepared panel": prepared_panel}), "panel_model_suite.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with b2:
            st.download_button("Download panel publication bundle", panel_publication_bundle(panel_coef, panel_fit, hausman), "panel_publication_bundle.zip", "application/zip")


elif page == "9. Detailed Greece GIS":
    module_guide("Map Greek NUTS-2/NUTS-3 outcomes and test spatial concentration.", "Select geography, identifier, metric and aggregation; compare colour and monochrome maps and inspect Moran/LISA diagnostics.", "A choropleth shows spatial pattern, not mechanism. Moran significance depends on the spatial-neighbour definition and permutation design.")
    st.subheader("Official Greek boundaries and spatial diagnostics")
    level = st.selectbox("Geography", list(GISCO_URLS))
    custom = st.file_uploader("Optional custom GeoJSON (e.g., municipalities/LAU)", type=["geojson", "json"])
    geography_candidates = list(df.columns)
    default_geo = next((i for i, c in enumerate(geography_candidates) if "region" in c.casefold() or "nuts" in c.casefold()), 0)
    geography = st.selectbox("Geographical identifier", geography_candidates, index=default_geo)
    metric = st.selectbox("Map metric", numeric)
    aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count", "Minimum", "Maximum"])
    try:
        geojson = cached_geojson(level, custom.getvalue() if custom else None)
        mapped = aggregate_geography(df, geography, metric, aggregation, level)
        colour, mono = st.tabs(["Colour", "Black & white"])
        with colour:
            figure_with_downloads(choropleth_figure(mapped, geojson, metric, False), "greece_map_colour_interactive", mapped, "This colour map shows the selected regional aggregation; compare totals with rates before interpreting regional performance.")
        with mono:
            figure_with_downloads(choropleth_figure(mapped, geojson, metric, True), "greece_map_bw_interactive", mapped, "This monochrome version carries the same evidence and is suitable for grayscale publication.")
        guided_dataframe("Mapped Greek geography", mapped)
        if level.startswith("NUTS 2"):
            global_m, local_m = moran_diagnostics(mapped, metric, permutations=999)
            st.subheader("Moran and LISA diagnostics")
            guided_dataframe("Global Moran diagnostics", global_m)
            guided_dataframe("Local spatial diagnostics", local_m)
            for comment in map_commentary(global_m, local_m, metric): st.info(comment)
        else:
            global_m, local_m = pd.DataFrame(), pd.DataFrame()
            st.caption("NUTS-3/custom local spatial diagnostics require centroids in the uploaded data; the detailed choropleth remains available.")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.download_button("Colour PNG (600 dpi)", static_map_bytes(mapped, geojson, metric, False, "png"), "greece_map_colour_600dpi.png", "image/png")
        with col2: st.download_button("B&W PNG (600 dpi)", static_map_bytes(mapped, geojson, metric, True, "png"), "greece_map_bw_600dpi.png", "image/png")
        with col3: st.download_button("Colour vector PDF", static_map_bytes(mapped, geojson, metric, False, "pdf"), "greece_map_colour.pdf", "application/pdf")
        with col4: st.download_button("B&W vector SVG", static_map_bytes(mapped, geojson, metric, True, "svg"), "greece_map_bw.svg", "image/svg+xml")
        st.download_button("Download complete GIS analytical tables", to_excel_bytes({"Mapped geography": mapped, "Global Moran": global_m, "Local spatial diagnostics": local_m}), "greece_gis_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        st.error(f"Map unavailable: {exc}")
        st.info("Check internet access to Eurostat GISCO or upload a GeoJSON boundary file. NUTS-2 region aliases are recognised in Greek and English.")


elif page == "10. Time series & multivariate":
    module_guide("Analyse temporal dependence, forecasts, latent dimensions, clusters and scale reliability.", "Use the tabs for the specific analytical question and export the displayed tables after checking sample adequacy.", "These methods answer different questions; do not infer causality from Granger predictability, clusters or PCA loadings.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stationarity", "Forecast & Granger", "PCA", "Clustering", "Scale reliability"])
    with tab1:
        time_col = st.selectbox("Time column", list(df.columns))
        variables = st.multiselect("Series", numeric, default=numeric[:min(5, len(numeric))], key="ts_vars")
        if st.button("Run ADF and KPSS"):
            st.session_state["ts_tests"] = time_series_tests(df, variables, time_col)
        guided_dataframe("Stationarity diagnostics", st.session_state.get("ts_tests", pd.DataFrame()))
    with tab2:
        time_col2 = st.selectbox("Ordered time variable", list(df.columns), key="forecast_time")
        series = st.selectbox("Forecast series", numeric, key="forecast_series")
        p = st.slider("AR order (p)", 0, 6, 1); d_order = st.slider("Difference order (d)", 0, 2, 1); q_order = st.slider("MA order (q)", 0, 6, 0)
        steps = st.slider("Forecast steps", 1, 24, 5)
        if st.button("Fit ARIMA"):
            try: st.session_state["arima"] = arima_forecast(df, series, time_col2, (p, d_order, q_order), steps)
            except Exception as exc: st.error(str(exc))
        if "arima" in st.session_state:
            forecast, forecast_fit = st.session_state["arima"]
            guided_dataframe("ARIMA model fit", forecast_fit)
            guided_dataframe("ARIMA forecast", forecast)
            arima_fig = px.line(forecast, x="forecast_step", y="forecast", markers=True, title="ARIMA forecast")
            arima_fig.add_scatter(x=forecast.forecast_step, y=forecast.ci_95_low, mode="lines", line=dict(width=0), showlegend=False)
            arima_fig.add_scatter(x=forecast.forecast_step, y=forecast.ci_95_high, mode="lines", fill="tonexty", line=dict(width=0), name="95% interval")
            figure_with_downloads(arima_fig, "arima_forecast", forecast, "Forecast intervals widen with horizon and are conditional on the selected ARIMA order and stable data-generating process.")
        cause = st.selectbox("Potential Granger cause", numeric, key="granger_cause")
        effect = st.selectbox("Granger effect", [c for c in numeric if c != cause], key="granger_effect")
        max_lag = st.slider("Maximum lag", 1, 12, 4)
        if st.button("Run Granger tests"):
            try: st.session_state["granger"] = granger_table(df, cause, effect, time_col2, max_lag)
            except Exception as exc: st.error(str(exc))
        guided_dataframe("Granger predictability tests", st.session_state.get("granger", pd.DataFrame()))
        st.caption("Granger predictability is not structural causality; stationarity and lag adequacy must be checked first.")
    with tab3:
        pca_vars = st.multiselect("PCA variables", numeric, default=numeric[:min(10, len(numeric))], key="pca_vars")
        components = st.slider("Components", 2, min(20, max(2, len(pca_vars))), min(5, max(2, len(pca_vars)))) if pca_vars else 2
        if st.button("Run PCA", disabled=len(pca_vars) < 2):
            st.session_state["pca"] = pca_table(df, pca_vars, components)
        if "pca" in st.session_state:
            loadings, variance = st.session_state["pca"]
            guided_dataframe("PCA explained variance", variance)
            guided_dataframe("PCA loadings", loadings)
            pca_fig = px.bar(variance, x="component", y="explained_variance_ratio", title="PCA explained variance")
            figure_with_downloads(pca_fig, "pca_explained_variance", variance, "Explained variance measures compression, not substantive importance. Interpret components from their loading patterns.")
    with tab4:
        cluster_vars = st.multiselect("Clustering variables", numeric, default=numeric[:min(8, len(numeric))], key="cluster_vars")
        k = st.slider("Clusters", 2, 12, 4)
        if st.button("Run k-means", disabled=len(cluster_vars) < 2):
            st.session_state["clusters"] = cluster_table(df, cluster_vars, k)
        if "clusters" in st.session_state:
            assignments, profiles = st.session_state["clusters"]
            guided_dataframe("Cluster profiles", profiles)
            guided_dataframe("Cluster assignments", assignments, max_rows=500)
    with tab5:
        items = st.multiselect("Scale items", numeric, default=numeric[:min(6, len(numeric))], key="alpha_items")
        if st.button("Calculate Cronbach's alpha", disabled=len(items) < 2):
            try: st.session_state["alpha"] = cronbach_alpha(df, items)
            except Exception as exc: st.error(str(exc))
        if "alpha" in st.session_state:
            alpha_summary, item_stats = st.session_state["alpha"]
            guided_dataframe("Reliability summary", alpha_summary)
            guided_dataframe("Item diagnostics", item_stats)
    multivariate_tables = {}
    if isinstance(st.session_state.get("ts_tests"), pd.DataFrame): multivariate_tables["Stationarity"] = st.session_state["ts_tests"]
    if "arima" in st.session_state:
        multivariate_tables["ARIMA forecast"], multivariate_tables["ARIMA fit"] = st.session_state["arima"]
    if isinstance(st.session_state.get("granger"), pd.DataFrame): multivariate_tables["Granger"] = st.session_state["granger"]
    if "pca" in st.session_state:
        multivariate_tables["PCA loadings"], multivariate_tables["PCA variance"] = st.session_state["pca"]
    if "clusters" in st.session_state:
        multivariate_tables["Cluster assignments"], multivariate_tables["Cluster profiles"] = st.session_state["clusters"]
    if "alpha" in st.session_state:
        multivariate_tables["Reliability summary"], multivariate_tables["Item diagnostics"] = st.session_state["alpha"]
    multivariate_tables = {k: v for k, v in multivariate_tables.items() if isinstance(v, pd.DataFrame) and not v.empty}
    if multivariate_tables:
        st.download_button("Download all time-series and multivariate tables", to_excel_bytes(multivariate_tables), "time_series_multivariate_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


elif page == "10A. Advanced clustering & segmentation":
    st.subheader("Advanced clustering, typologies and absorption segmentation")
    module_guide(
        "Discover reproducible project or regional typologies using one or many numeric variables, including clustering based solely on resource absorption.",
        "Select variables, algorithm and scaling. Use automatic k for K-means, hierarchical or Gaussian mixture; use eps/min_samples for DBSCAN. Inspect separation diagnostics and substantive profiles before naming clusters.",
        "Clusters are descriptive partitions. They require stability checks and substantive validation; they do not reveal causal mechanisms or objectively true categories.",
    )
    absorption_candidates = [c for c in numeric if "absorp" in c.casefold() or "απορ" in c.casefold()]
    default_cluster_vars = absorption_candidates[:1] or numeric[:min(3, len(numeric))]
    cluster_vars = st.multiselect("Clustering variables — one variable is allowed", numeric, default=default_cluster_vars, max_selections=100, key="advanced_cluster_vars")
    c1, c2, c3 = st.columns(3)
    with c1: cluster_method = st.selectbox("Algorithm", ["K-means", "Hierarchical agglomerative", "Gaussian mixture", "DBSCAN"], key="advanced_cluster_method")
    with c2: cluster_scaling = st.selectbox("Scaling", ["Standard (z-score)", "Robust (median/IQR)", "Min-max [0,1]", "None"], key="advanced_cluster_scaling")
    with c3: cluster_seed = st.number_input("Reproducible seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="advanced_cluster_seed")
    cluster_method_notes = {
        "K-means": "Best for compact, approximately spherical clusters; fast and transparent but sensitive to scale and outliers.",
        "Hierarchical agglomerative": "Builds nested similarity groups and is useful for typology exploration; linkage choice changes the geometry.",
        "Gaussian mixture": "A probabilistic latent-class model allowing elliptical clusters; BIC/AIC are supplied alongside separation metrics.",
        "DBSCAN": "Density-based clustering that discovers irregular shapes and labels sparse observations as noise; highly sensitive to eps and min_samples.",
    }
    st.caption(cluster_method_notes[cluster_method])
    automatic_k = False
    cluster_k, cluster_max_k, linkage, db_eps, db_min = 4, 10, "ward", .7, 5
    if cluster_method != "DBSCAN":
        d1, d2, d3 = st.columns(3)
        with d1: automatic_k = st.checkbox("Automatically select k by silhouette", value=True, key="advanced_auto_k")
        with d2: cluster_k = st.slider("Requested clusters (k)", 2, 20, 4, key="advanced_cluster_k", disabled=automatic_k)
        with d3: cluster_max_k = st.slider("Maximum k to evaluate", 2, 20, 10, key="advanced_cluster_max_k", disabled=not automatic_k)
        if cluster_method == "Hierarchical agglomerative":
            linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="advanced_linkage")
    else:
        d1, d2 = st.columns(2)
        with d1: db_eps = st.slider("Neighbourhood radius (eps)", .05, 5.0, .70, .05, key="dbscan_eps")
        with d2: db_min = st.slider("Minimum neighbouring observations", 2, 100, 5, key="dbscan_min_samples")
    if len(cluster_vars) == 1:
        st.success(f"One-dimensional clustering enabled for: {cluster_vars[0]}. This directly supports typologies based solely on resource absorption.")
    if st.button("Run advanced clustering", type="primary", disabled=not cluster_vars):
        try:
            with st.spinner("Fitting clusters, evaluating separation and constructing publication profiles…"):
                st.session_state["advanced_clusters"] = advanced_clustering(df, cluster_vars, cluster_method, cluster_k, cluster_scaling, automatic_k, cluster_max_k, linkage, db_eps, db_min, int(cluster_seed))
        except Exception as exc:
            st.error(f"Advanced clustering failed: {exc}")
    if "advanced_clusters" in st.session_state:
        clustering = st.session_state["advanced_clusters"]
        selected_diag = clustering.diagnostics.loc[clustering.diagnostics.selected].iloc[0]
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Clusters", f"{int(selected_diag.clusters):,}")
        q2.metric("Observations", f"{len(clustering.assignments):,}")
        q3.metric("Silhouette", "—" if pd.isna(selected_diag.silhouette) else f"{selected_diag.silhouette:.3f}")
        q4.metric("Noise points", f"{int(selected_diag.noise_observations):,}")
        q5.metric("Stability ARI", f"{selected_diag.perturbation_stability_ari:.3f}")
        table_with_downloads("Cluster selection diagnostics", clustering.diagnostics, "advanced_cluster_diagnostics", "Higher silhouette and Calinski–Harabasz are preferable; lower Davies–Bouldin is preferable. They are diagnostic aids, not substitutes for theory.")
        table_with_downloads("Substantive cluster profiles", clustering.profiles, "advanced_cluster_profiles", "Means, medians, counts and sample shares provide the evidence needed to name each cluster transparently.")
        st.subheader("Interpretive assessment")
        for comment in clustering.interpretation: st.info(comment)
        if len(clustering.diagnostics) > 1:
            diagnostic_fig = px.line(clustering.diagnostics, x="clusters", y="silhouette", markers=True, title="Cluster separation across candidate k values")
            diagnostic_fig.add_vline(x=int(selected_diag.clusters), line_dash="dash", line_color="#D9A441")
            figure_with_downloads(diagnostic_fig, "cluster_k_diagnostics", clustering.diagnostics, "The selected k maximises silhouette over the evaluated range; adjacent solutions should still be inspected for substantive stability.")
        projection_fig = px.scatter(clustering.embedding, x="dimension_1", y="dimension_2", color="cluster_label", symbol="cluster_label", hover_data=["row_index"], title="Cluster projection / one-dimensional segmentation")
        figure_with_downloads(projection_fig, "advanced_cluster_projection", clustering.embedding, str(clustering.embedding.projection_note.iloc[0]))
        mean_cols = [c for c in clustering.profiles if c.startswith("mean_")]
        if mean_cols:
            profile_matrix = clustering.profiles.set_index("cluster_label")[mean_cols].copy()
            profile_matrix.columns = [c.removeprefix("mean_") for c in profile_matrix.columns]
            z = (profile_matrix - profile_matrix.mean()) / profile_matrix.std(ddof=0).replace(0, np.nan)
            z = z.fillna(0)
            profile_fig = px.imshow(z, aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, title="Standardised cluster profile heatmap")
            figure_with_downloads(profile_fig, "advanced_cluster_profiles", z.reset_index(), "Positive cells indicate above-cluster-average values for that variable; negative cells indicate below-average values.")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Download complete clustering workbook", to_excel_bytes({"Diagnostics": clustering.diagnostics, "Profiles": clustering.profiles, "Assignments": clustering.assignments, "Projection": clustering.embedding}), "advanced_clustering_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with e2:
            st.download_button("Download clustering publication bundle", clustering_publication_bundle(clustering.assignments, clustering.profiles, clustering.embedding, clustering.diagnostics), "clustering_publication_bundle.zip", "application/zip")


elif page == "10B. Predictive model laboratory":
    st.subheader("Cross-validated predictive modelling and driver importance")
    module_guide(
        "Compare transparent linear models with nonlinear ensembles using out-of-fold evidence rather than in-sample fit.",
        "Choose a continuous outcome and up to 200 numeric predictors, select 3–10 folds and run the comparison. Inspect RMSE, MAE, R², out-of-fold predictions and permutation importance.",
        "Predictive accuracy and variable importance do not establish causality. Grouped regional or time-dependent records require grouped or time-ordered validation for final claims.",
    )
    pred_y = st.selectbox("Continuous outcome", numeric, key="predictive_y")
    pred_candidates = [c for c in numeric if c != pred_y]
    pred_x = st.multiselect("Predictors (maximum 200)", pred_candidates, default=pred_candidates[:min(12, len(pred_candidates))], max_selections=200, key="predictive_x")
    m1, m2 = st.columns(2)
    with m1: pred_folds = st.slider("Cross-validation folds", 3, 10, 5, key="predictive_folds")
    with m2: pred_seed = st.number_input("Validation seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="predictive_seed")
    st.caption("Candidate models: OLS, Ridge, Lasso, Elastic Net, random forest, extra trees and gradient boosting. Missing predictors are median-imputed within each model pipeline.")
    if st.button("Run predictive comparison", type="primary", disabled=not pred_x):
        try:
            with st.spinner("Running cross-validation across seven model families…"):
                st.session_state["predictive_suite"] = predictive_model_comparison(df, pred_y, pred_x, pred_folds, int(pred_seed))
        except Exception as exc:
            st.error(f"Predictive comparison failed: {exc}")
    if "predictive_suite" in st.session_state:
        performance, predictions, importance, pred_comments = st.session_state["predictive_suite"]
        best = performance.iloc[0]
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Best model", best.model)
        u2.metric("CV RMSE", f"{best.cross_validated_rmse:.4g}")
        u3.metric("CV MAE", f"{best.cross_validated_mae:.4g}")
        u4.metric("CV R²", f"{best.cross_validated_r_squared:.3f}")
        table_with_downloads("Cross-validated model performance", performance, "predictive_model_performance", "All metrics use out-of-fold predictions; lower RMSE/MAE and higher R² are preferable.")
        table_with_downloads("Permutation predictor importance", importance, "predictive_permutation_importance", "Importance is the deterioration in predictive score after a variable is permuted; correlated predictors can share or mask importance.")
        st.subheader("Predictive interpretation")
        for comment in pred_comments: st.info(comment)
        perf_fig = px.bar(performance.sort_values("cross_validated_rmse"), x="cross_validated_rmse", y="model", orientation="h", color="cross_validated_r_squared", title="Cross-validated model performance", color_continuous_scale="Teal")
        figure_with_downloads(perf_fig, "predictive_model_performance", performance, "RMSE is expressed in outcome units. Negative cross-validated R² means the model predicts worse than the fold-specific mean benchmark.")
        best_name = str(best.model)
        pred_fig = px.scatter(predictions, x="observed", y=best_name, trendline="ols", title=f"Out-of-fold observed versus predicted: {best_name}")
        figure_with_downloads(pred_fig, "predictive_observed_vs_predicted", predictions[["row_index", "observed", best_name]], "Because every point is predicted while excluded from model fitting, this figure is more honest than an in-sample fitted plot.")
        imp_show = importance.head(30).sort_values("permutation_importance_mean")
        imp_fig = px.bar(imp_show, x="permutation_importance_mean", y="variable", orientation="h", error_x="permutation_importance_sd", title="Predictor importance in the best model")
        figure_with_downloads(imp_fig, "predictive_variable_importance", imp_show, "Importance near or below zero indicates little stable predictive contribution under this validation design.")
        v1, v2 = st.columns(2)
        with v1:
            st.download_button("Download complete predictive workbook", to_excel_bytes({"Performance": performance, "Predictions": predictions, "Importance": importance}), "predictive_model_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with v2:
            st.download_button("Download predictive publication bundle", predictive_publication_bundle(performance, importance, predictions), "predictive_publication_bundle.zip", "application/zip")


elif page == "11. Publication figures & HTML report":
    module_guide("Create reusable, publication-ready visual and narrative outputs.", "Choose the chart structure, title and grouping; preview it and download the complete colour/black-and-white package or HTML report.", "The package includes 600-dpi raster and vector files plus plotted data so every figure can be audited and reproduced.")
    st.subheader("Interactive preview and publication package")
    chart = st.selectbox("Chart", ["Bar", "Line", "Scatter", "Box", "Histogram"])
    x_options = list(df.columns)
    x = st.selectbox("X / category / time", x_options)
    y = st.selectbox("Y", numeric)
    group = st.selectbox("Group/series (optional)", [None] + [c for c in categorical if c != x])
    title = st.text_input("Figure title", value=f"{y} by {x}")
    try:
        fig = interactive_figure(df, chart, x, y, group, title)
        figure_with_downloads(fig, "publication_figure_preview", df[[c for c in [x, y, group] if c is not None]], "This preview uses the same selected variables as the downloadable colour and black-and-white publication package.")
        st.download_button("Download publication package (colour + B&W; PNG 600 dpi + SVG + PDF + data)", publication_bundle(df, chart, x, y, group, title), "publication_figure_bundle.zip", "application/zip")
        st.subheader("Portable HTML results report")
        include_desc = st.checkbox("Include descriptive table", value=True)
        tables = {"Data quality": quality_summary(df)}
        comments = []
        if include_desc: tables["Descriptive statistics"] = descriptive_statistics(df, numeric[:100])
        if st.session_state.get("model_output"):
            out = st.session_state["model_output"]
            tables.update({"Model fit": out.fit, "Coefficients": out.coefficients, "Diagnostics": out.diagnostics})
            comments.extend(out.interpretation)
        if "mc_ols" in st.session_state:
            mc_summary, _, mc_fit = st.session_state["mc_ols"]
            tables.update({"Monte Carlo OLS settings": mc_fit, "Monte Carlo coefficient uncertainty": mc_summary})
            comments.append("Monte Carlo intervals are conditional on the selected bootstrap/data-generating mechanism and reproducible seed.")
        if "mc_portfolio" in st.session_state:
            port_summary, projects, _ = st.session_state["mc_portfolio"]
            tables.update({"Portfolio uncertainty": port_summary, "Project selection probabilities": projects})
        if "panel_suite" in st.session_state:
            panel_fit, panel_coef, hausman, _, panel_comments = st.session_state["panel_suite"]
            tables.update({"Panel model fit": panel_fit, "Panel coefficients": panel_coef, "Hausman test": hausman})
            comments.extend(panel_comments)
        if "advanced_clusters" in st.session_state:
            clustering = st.session_state["advanced_clusters"]
            tables.update({"Cluster diagnostics": clustering.diagnostics, "Cluster profiles": clustering.profiles})
            comments.extend(clustering.interpretation)
        if "predictive_suite" in st.session_state:
            performance, _, importance, pred_comments = st.session_state["predictive_suite"]
            tables.update({"Predictive performance": performance, "Predictor importance": importance})
            comments.extend(pred_comments)
        if "mcda_output" in st.session_state:
            mcda_out = st.session_state["mcda_output"]
            tables.update({
                "MCDA rankings": mcda_out.rankings,
                "MCDA criterion weights": mcda_out.weights,
                "MCDA sensitivity": mcda_out.sensitivity,
                "MCDA rank acceptability": mcda_out.acceptability_summary,
            })
            comments.extend(mcda_out.interpretation)
        if not comments:
            comments = ["This report currently contains descriptive evidence. Run analytical modules to append model, simulation, clustering, panel or predictive results automatically."]
        report = build_html_report("Makryvelios Research Analytics Report", f"Active dataset: {selected_label}; {len(df):,} records; {df.shape[1]:,} variables.", tables, [fig.to_html(full_html=False, include_plotlyjs=True)], comments)
        r1, r2 = st.columns(2)
        with r1: st.download_button("Download self-contained HTML report", report, "makryvelios_analytics_report.html", "text/html")
        with r2: st.download_button("Download consolidated results workbook", to_excel_bytes(tables), "makryvelios_consolidated_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        st.error(str(exc))


elif page == "12. Scenario & allocation engine":
    module_guide("Translate estimated associations into transparent what-if scenarios and constrained allocations.", "Set the outcome, driver or allocation score, state the resource envelope and run the chosen tab.", "Scenario outputs are conditional projections. Policy use requires defensible causal assumptions and explicit fairness/eligibility constraints.")
    tab1, tab2 = st.tabs(["Econometric shock", "Constrained allocation"])
    with tab1:
        outcome = st.selectbox("Outcome", numeric, key="sc_y")
        driver = st.selectbox("Driver", [c for c in numeric if c != outcome], key="sc_driver")
        controls = st.multiselect("Controls", [c for c in numeric if c not in {outcome, driver}], default=[c for c in numeric if c not in {outcome, driver}][:min(5, len(numeric))], key="sc_controls")
        shock = st.slider("Driver shock (%)", -50, 100, 10, 5) / 100
        segment = st.selectbox("Summarise by", [None] + categorical, key="sc_segment")
        if st.button("Run shock scenario"):
            try:
                model = fit_detailed_model(df, outcome, [driver] + controls, estimator="OLS", covariance="HC3")
                beta = float(model.coefficients.loc[model.coefficients.term == driver, "coefficient"].iloc[0])
                base_driver = pd.to_numeric(df[driver], errors="coerce")
                delta = beta * base_driver * shock
                scenario = pd.DataFrame({"row_index": df.index, "baseline_driver": base_driver, "driver_shock": base_driver * shock, "predicted_outcome_delta": delta})
                if segment:
                    scenario[segment] = df[segment]
                    summary = scenario.groupby(segment, dropna=False).agg(observations=("predicted_outcome_delta", "size"), mean_predicted_delta=("predicted_outcome_delta", "mean"), total_predicted_delta=("predicted_outcome_delta", "sum")).reset_index()
                else:
                    summary = pd.DataFrame([{"observations": len(scenario), "mean_predicted_delta": delta.mean(), "total_predicted_delta": delta.sum()}])
                st.session_state["scenario"] = (summary, scenario, beta)
            except Exception as exc:
                st.error(str(exc))
        if "scenario" in st.session_state:
            summary, scenario, beta = st.session_state["scenario"]
            guided_dataframe("Scenario summary", summary)
            st.info(f"Estimated partial slope for {driver}: {beta:.5g}. The scenario is a ceteris-paribus linear projection, not a causal forecast.")
            if segment:
                shock_fig = px.bar(summary.sort_values("mean_predicted_delta"), x="mean_predicted_delta", y=segment, orientation="h", title="Mean projected outcome change")
                figure_with_downloads(shock_fig, "scenario_mean_change", summary, "Projected changes inherit the linear OLS association and the ceteris-paribus assumption; they are not causal forecasts.")
            st.download_button("Download scenario tables", to_excel_bytes({"Summary": summary, "Row-level": scenario}), "shock_scenario.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with tab2:
        group_col = st.selectbox("Allocation unit", categorical, key="alloc_group") if categorical else None
        current_col = st.selectbox("Current allocation", numeric, key="alloc_current")
        score_col = st.selectbox("Need/benefit score", [c for c in numeric if c != current_col], key="alloc_score")
        total_extra = st.number_input("Additional resource envelope", min_value=0.0, value=1_000_000.0, step=100_000.0)
        max_share = st.slider("Maximum share for any unit (%)", 5, 100, 30) / 100
        if st.button("Optimise allocation", disabled=group_col is None):
            agg = df.groupby(group_col).agg(current=(current_col, "sum"), score=(score_col, "mean")).dropna().reset_index()
            scores = agg.score.to_numpy(float)
            if scores.max() != scores.min(): scores = (scores - scores.min()) / (scores.max() - scores.min())
            result = linprog(-scores, A_eq=np.ones((1, len(agg))), b_eq=[total_extra], bounds=[(0, total_extra * max_share)] * len(agg), method="highs")
            if result.success:
                agg["additional_allocation"] = result.x; agg["new_total"] = agg.current + result.x; agg["share_of_extra"] = agg.additional_allocation / total_extra if total_extra else 0
                st.session_state["allocation"] = agg
            else: st.error(result.message)
        allocation = st.session_state.get("allocation", pd.DataFrame())
        if not allocation.empty:
            guided_dataframe("Optimised allocation", allocation.sort_values("additional_allocation", ascending=False))
            allocation_fig = px.bar(allocation.sort_values("additional_allocation"), x="additional_allocation", y=group_col, orientation="h", title="Optimised additional allocation")
            figure_with_downloads(allocation_fig, "optimised_allocation", allocation, "The result maximises the supplied score under the current envelope and cap; it does not encode omitted policy, fairness or eligibility rules.")
            st.warning("The optimisation maximises the supplied score subject only to the envelope and cap. Policy use requires explicit floors, eligibility rules, fairness constraints, costs and sensitivity analysis.")


elif page == "12A. Dedicated MCDA engine":
    module_guide(
        "Rank projects, regions or other alternatives against multiple economic, environmental, spatial, technical and policy criteria.",
        "Select the alternative identifier and criteria, define benefit/cost directions and weights, run several ranking methods, then inspect method agreement, weight sensitivity and Monte Carlo rank acceptability before downloading the complete evidence package.",
        "MCDA makes value judgements explicit; it does not discover a uniquely correct policy choice. Rankings are conditional on criterion definitions, data quality, directions, weights, compensability and the chosen preference model.",
    )
    st.subheader("Decision problem and criteria")
    id_candidates = [None] + list(df.columns)
    suggested_id = next((c for c in df.columns if any(token in c.lower() for token in ("project", "alternative", "application", "code", "id"))), None)
    id_index = id_candidates.index(suggested_id) if suggested_id in id_candidates else 0
    alternative_id = st.selectbox("Alternative identifier (optional)", id_candidates, index=id_index, key="mcda_id")
    default_criteria = numeric[:min(6, len(numeric))]
    criteria = st.multiselect("Decision criteria (maximum 50)", numeric, default=default_criteria, key="mcda_criteria")
    if len(criteria) > 50:
        st.error("Select no more than 50 criteria for one auditable MCDA run.")
    c1, c2, c3 = st.columns(3)
    with c1:
        missing_rule = st.selectbox("Missing-data rule", ["Median imputation", "Complete cases"], key="mcda_missing")
    with c2:
        weight_method = st.selectbox("Weighting method", list(WEIGHT_METHODS), key="mcda_weight_method")
    with c3:
        selected_methods = st.multiselect("Ranking methods", list(MCDA_METHODS), default=list(MCDA_METHODS), key="mcda_methods")

    config = pd.DataFrame({
        "criterion": criteria,
        "direction": ["Maximise"] * len(criteria),
        "weight": [1 / len(criteria)] * len(criteria) if criteria else [],
    })
    edited_config = st.data_editor(
        config,
        width="stretch",
        hide_index=True,
        disabled=["criterion"],
        column_config={
            "criterion": st.column_config.TextColumn("Criterion"),
            "direction": st.column_config.SelectboxColumn("Direction", options=["Maximise", "Minimise"], required=True),
            "weight": st.column_config.NumberColumn("User weight", min_value=0.0, format="%.5f"),
        },
        key="mcda_config_editor",
    )
    st.caption("Maximise means that larger values are preferred; Minimise is appropriate for cost, risk, delay, land use or emissions. User weights are normalised automatically and are used only when ‘User-defined’ is selected.")

    pairwise = None
    if weight_method == "AHP pairwise":
        with st.expander("AHP pairwise comparison matrix", expanded=True):
            st.markdown("Enter Saaty judgements above the diagonal: 1 = equal importance, 3 = moderate, 5 = strong, 7 = very strong and 9 = extreme. Reciprocals are reconstructed automatically. Keep the consistency ratio below approximately 0.10.")
            if len(criteria) > 15:
                st.error("AHP pairwise weighting is limited to 15 criteria because the number of judgements grows quadratically. Use Entropy or CRITIC for larger sets.")
            pairwise_template = pd.DataFrame(np.ones((len(criteria), len(criteria))), index=criteria, columns=criteria)
            pairwise = st.data_editor(pairwise_template, width="stretch", key="mcda_ahp_editor") if criteria else None

    st.subheader("Robustness design")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        primary_options = [m for m in selected_methods if m in {"MAVT", "TOPSIS"}] or ["TOPSIS"]
        primary_method = st.selectbox("Primary robustness method", primary_options, key="mcda_primary")
    with r2:
        sensitivity = st.slider("Weight perturbation (%)", 5, 75, 25, 5, key="mcda_sensitivity") / 100
    with r3:
        simulations = st.slider("Monte Carlo weight draws", 100, 5_000, 1_000, 100, key="mcda_simulations")
    with r4:
        concentration = st.slider("Weight concentration", 5, 500, 75, 5, key="mcda_concentration", help="Higher values keep simulated weights closer to the baseline weights.")
    seed = st.number_input("Reproducible random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="mcda_seed")

    ahp_disabled = weight_method == "AHP pairwise" and len(criteria) > 15
    if st.button("Run dedicated MCDA", type="primary", disabled=len(criteria) < 2 or len(criteria) > 50 or not selected_methods or ahp_disabled):
        try:
            directions = dict(zip(edited_config.criterion, edited_config.direction))
            custom_weights = dict(zip(edited_config.criterion, pd.to_numeric(edited_config.weight, errors="coerce").fillna(0)))
            output = mcda_analysis(
                df,
                criteria=list(edited_config.criterion),
                directions=directions,
                weight_method=weight_method,
                user_weights=custom_weights,
                pairwise_matrix=pairwise,
                methods=selected_methods,
                alternative_id=alternative_id,
                missing=missing_rule,
                primary_method=primary_method,
                simulations=simulations,
                concentration=concentration,
                sensitivity_range=sensitivity,
                seed=int(seed),
            )
            st.session_state["mcda_output"] = output
        except Exception as exc:
            st.error(str(exc))

    mcda_out = st.session_state.get("mcda_output")
    if mcda_out is not None:
        rank_col = f"{mcda_out.primary_method}_rank"
        score_col = f"{mcda_out.primary_method}_score"
        top = mcda_out.rankings.nsmallest(1, rank_col).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alternatives", f"{len(mcda_out.rankings):,}")
        m2.metric("Criteria", f"{len(mcda_out.weights):,}")
        m3.metric("Primary method", mcda_out.primary_method)
        m4.metric("Top alternative", str(top.alternative))
        table_with_downloads("MCDA ranking table", mcda_out.rankings, "mcda_rankings", "Each score and rank is method-specific. Compare the methods rather than reporting only the preferred result.")
        top_plot = mcda_out.rankings.nsmallest(min(30, len(mcda_out.rankings)), rank_col).sort_values(rank_col, ascending=False)
        rank_fig = px.bar(top_plot, x=score_col, y="alternative", orientation="h", color=rank_col, title=f"Leading alternatives — {mcda_out.primary_method}", color_continuous_scale="Teal")
        figure_with_downloads(rank_fig, "mcda_leading_alternatives", top_plot, "The position depends on the stated criteria, directions, weights and preference method; it is not a causal estimate or an automatic funding decision.")
        left, right = st.columns(2)
        with left:
            weight_fig = px.bar(mcda_out.weights.sort_values("weight"), x="weight", y="criterion", orientation="h", color="direction", title="Normalised criterion weights")
            figure_with_downloads(weight_fig, "mcda_criterion_weights", mcda_out.weights, "Objective weights describe dispersion or conflict in the observed sample; they do not replace substantive policy judgements.")
        with right:
            accept_show = mcda_out.acceptability_summary.head(30).sort_values("probability_rank_1")
            accept_fig = px.bar(accept_show, x="probability_rank_1", y="alternative", orientation="h", color="expected_rank", title="Monte Carlo first-rank probability", color_continuous_scale="Teal")
            figure_with_downloads(accept_fig, "mcda_rank_acceptability", accept_show, "Rank acceptability represents robustness to simulated criterion weights, not the probability of technical or commercial project success.")
        table_with_downloads("Criterion weights and directions", mcda_out.weights, "mcda_weights")
        table_with_downloads("Method rank correlations", mcda_out.rank_correlations, "mcda_method_correlations", "Low Spearman agreement means that the preference models encode materially different compensation or distance assumptions.")
        table_with_downloads("Weight sensitivity analysis", mcda_out.sensitivity, "mcda_weight_sensitivity", "Rank stability near one indicates limited reordering under the selected one-at-a-time weight perturbation.")
        table_with_downloads("Monte Carlo rank acceptability", mcda_out.acceptability_summary, "mcda_acceptability", "Probability of rank 1 and expected rank are calculated across reproducible Dirichlet weight draws.")
        st.subheader("MCDA interpretation and safeguards")
        for comment in mcda_out.interpretation:
            st.info(comment)
        workbook = to_excel_bytes({
            "Rankings": mcda_out.rankings,
            "Weights": mcda_out.weights,
            "Normalised matrix": mcda_out.normalised_matrix,
            "Method correlations": mcda_out.rank_correlations,
            "Sensitivity": mcda_out.sensitivity,
            "Acceptability summary": mcda_out.acceptability_summary,
            "Rank acceptability": mcda_out.rank_acceptability,
            "Diagnostics": mcda_out.diagnostics,
        })
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download complete MCDA workbook", workbook, "mcda_complete_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d2:
            st.download_button("Download MCDA publication bundle", mcda_publication_bundle(mcda_out), "mcda_publication_bundle.zip", "application/zip")


elif page == "12A.1 ITA / public-funding decision support":
    render_ita_module(df)


elif page == "12A.1B GAMS-compatible ITA Studio":
    render_gams_studio(df)

elif page == "12A.2 Expert respondent analytics":
    render_respondent_module(df, selected_label)


elif page == "12C. Frontier methods laboratory":
    render_frontier_methods(df)


elif page == "12D. Agentic Research Mode":
    render_agentic_research(df, selected_label)


elif page == "12B. Research Command Chair":
    st.subheader("Research Command Chair — free/offline evidence and protocol assistant")
    module_guide(
        "Give the dashboard a research question, algorithm, equation, steps and limitations; restrict the XLSX/CSV and PDF evidence to an exact analytical scope; receive reproducible tables, natural-language interpretation and a paper-writing blueprint.",
        "Define the data scope and years, select PDF pages or keywords, document the protocol, run it, then ask evidence-grounded questions and download the complete research bundle.",
        "The built-in interpreter is deterministic and free. Optional local Ollama or the user-key LLM co-pilot can improve interpretation/prose after computation. No language model can turn an observational association into a causal effect or validate an unsupported custom algorithm.",
    )
    st.info("Privacy-first default: spreadsheet and PDF processing remains inside the running app and no paid API key is required. An external LLM receives only a compact computed-evidence summary when the user explicitly selects the External LLM engine and presses Ask; every core function and export remains available offline.")

    pdf_uploads = st.file_uploader(
        "Upload one or many supporting PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="chair_pdf_uploads",
        help="PDFs are indexed page-by-page locally. Scanned image-only PDFs require OCR before text can be analysed.",
    )
    if pdf_uploads:
        try:
            pdf_pages = parse_pdf_payloads(tuple((item.name, item.getvalue()) for item in pdf_uploads))
        except Exception as exc:
            st.error(f"PDF extraction failed: {exc}")
            pdf_pages = pd.DataFrame(columns=["document", "page", "text", "characters"])
    else:
        pdf_pages = pd.DataFrame(columns=["document", "page", "text", "characters"])

    quick_tab, scope_tab, protocol_tab, pdf_tab, answer_tab, prompt_tab = st.tabs([
        "Start here · Guided chat autopilot",
        "Advanced 1 · Select XLSX/CSV evidence", "Advanced 2 · Algorithm & equation",
        "Advanced 3 · Select PDF evidence", "Advanced 4 · Results & paper report",
        "Question library",
    ])

    with scope_tab:
        st.markdown("#### Exact analytical scope")
        if df.empty:
            st.warning("No spreadsheet dataset is active. PDF-only evidence can still be selected, but statistical protocols require an XLSX/CSV dataset.")
            chair_columns: list[str] = []
            chair_year = None
            chair_start = chair_end = None
            categorical_filters: dict[str, list] = {}
            numeric_filters: dict[str, tuple[float, float]] = {}
            scoped_preview = pd.DataFrame()
        else:
            chair_columns = st.multiselect(
                "Variables to retain (up to 1,000)",
                list(df.columns),
                default=list(df.columns)[:min(30, len(df.columns))],
                max_selections=1000,
                key="chair_columns",
                help="Only selected variables enter the Research Chair results and exports. The original active dataset remains unchanged.",
            )
            named_years = [c for c in df.columns if any(token in str(c).lower() for token in ["year", "date", "έτος", "ετος"])]
            chair_year = st.selectbox("Year/date variable (optional)", [None] + named_years + [c for c in df.columns if c not in named_years], key="chair_year")
            bounds = year_bounds(df, chair_year)
            if bounds and bounds[0] < bounds[1]:
                chair_start, chair_end = st.slider("Years to analyse", bounds[0], bounds[1], (bounds[0], bounds[1]), key="chair_years")
            elif bounds:
                chair_start = chair_end = bounds[0]
                st.caption(f"Only one valid year is present: {bounds[0]}.")
            else:
                chair_start = chair_end = None
                if chair_year:
                    st.caption("The selected variable could not be interpreted as years between 1800 and 2200; no year restriction will be applied.")
            filter_columns = st.multiselect("Additional row filters (maximum 6)", list(df.columns), max_selections=6, key="chair_filter_columns")
            categorical_filters = {}
            numeric_filters = {}
            for position, column in enumerate(filter_columns):
                series = df[column]
                if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                    low, high = float(series.min()), float(series.max())
                    c1, c2 = st.columns(2)
                    with c1:
                        selected_low = st.number_input(f"{column}: minimum", value=low, key=f"chair_num_low_{position}")
                    with c2:
                        selected_high = st.number_input(f"{column}: maximum", value=high, key=f"chair_num_high_{position}")
                    numeric_filters[column] = (float(selected_low), float(selected_high))
                else:
                    options = list(series.dropna().astype(str).value_counts().head(250).index)
                    chosen_values = st.multiselect(f"{column}: retained values", options, default=options, key=f"chair_cat_{position}")
                    categorical_filters[column] = chosen_values
            try:
                filter_df = df.copy()
                for column, values in categorical_filters.items():
                    filter_df[column] = filter_df[column].astype(str)
                scoped_preview = apply_scope(filter_df, chair_columns, chair_year, chair_start, chair_end, categorical_filters, numeric_filters)
            except Exception as exc:
                st.error(f"Scope could not be applied: {exc}")
                scoped_preview = pd.DataFrame()
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected records", f"{len(scoped_preview):,}")
            m2.metric("Selected variables", f"{scoped_preview.shape[1]:,}")
            retained = (100 * len(scoped_preview) / len(df)) if len(df) else 0
            m3.metric("Rows retained", f"{retained:.1f}%")
            guided_dataframe("Selected analytical scope preview", scoped_preview, max_rows=250)
            st.download_button("Download current filtered scope", scoped_preview.to_csv(index=False).encode("utf-8-sig"), "research_chair_filtered_scope.csv", "text/csv")

    with protocol_tab:
        st.markdown("#### Research command")
        working_title = st.text_input("Working paper title", value="Evidence-grounded research paper", key="chair_title")
        research_question = st.text_area("Principal research question", placeholder="Example: How has regional resource absorption changed over time, and which project characteristics are associated with it?", key="chair_question")
        algorithm = st.selectbox(
            "Algorithm to execute",
            ["Descriptive profile", "Longitudinal trend", "Correlation screening", "OLS specification", "Custom documented algorithm"],
            key="chair_algorithm",
        )
        available_numeric = list(scoped_preview.select_dtypes(include=np.number).columns) if not scoped_preview.empty else []
        outcome = st.selectbox("Outcome / principal measure", [None] + available_numeric, key="chair_outcome")
        predictors = st.multiselect("Predictors / additional measures", [c for c in available_numeric if c != outcome], max_selections=1000, key="chair_predictors")
        group_candidates = [c for c in scoped_preview.columns if c not in available_numeric] if not scoped_preview.empty else []
        group_column = st.selectbox("Grouping variable (optional)", [None] + group_candidates, key="chair_group")
        aggregation = st.selectbox("Longitudinal aggregation", ["Mean", "Sum", "Median", "Count"], key="chair_aggregation")
        equation = st.text_area(
            "Equation for the report (LaTeX)",
            value=r"Y_{it} = \beta_0 + \beta_1 X_{it} + \mu_i + \lambda_t + \varepsilon_{it}",
            key="chair_equation",
            help="This equation is rendered and documented. It is not automatically treated as executable code.",
        )
        if equation.strip():
            try: st.latex(equation)
            except Exception: st.code(equation)
        st.markdown("##### Optional safe derived variable")
        d1, d2 = st.columns([1, 2])
        with d1:
            derived_name = st.text_input("New variable name", placeholder="absorption_intensity", key="chair_derived_name")
        with d2:
            derived_expression = st.text_input("Computable expression", placeholder="Public_Expenditure / Project_Budget", key="chair_derived_expression")
        derived_role = st.selectbox("Role of the derived variable", ["Document only", "Use as outcome", "Add as predictor"], key="chair_derived_role")
        st.caption("Allowed: numeric column names, constants, parentheses, + − × ÷, powers, %, log, log1p, exp, sqrt and abs. Arbitrary Python and file/system commands are blocked.")
        steps = st.text_area("Required algorithmic steps", value="1. Define the analytical sample.\n2. Audit missingness and units.\n3. Estimate the primary specification.\n4. Inspect uncertainty and diagnostics.\n5. Run robustness checks.\n6. Interpret only within the declared limitations.", height=165, key="chair_steps")
        limitations = st.text_area("Known limitations and prohibited interpretations", placeholder="Example: observational data; regional aggregates; no causal language without an identification strategy; potential measurement error.", height=125, key="chair_limitations")
        researcher_notes = st.text_area("Additional notes or instructions", placeholder="Paste methodological notes, supervisor comments or journal requirements.", key="chair_notes")

    with pdf_tab:
        st.markdown("#### Documentary evidence scope")
        if pdf_pages.empty:
            selected_documents: list[str] = []
            page_ranges: dict[str, tuple[int, int]] = {}
            pdf_keywords = ""
            selected_evidence = pdf_pages.copy()
            st.info("Upload PDFs above to select exact documents, page ranges and keyword-matching passages.")
        else:
            inventory = pdf_pages.groupby("document", as_index=False).agg(pages=("page", "max"), extracted_characters=("characters", "sum"))
            guided_dataframe("PDF evidence inventory", inventory)
            document_options = list(inventory.document)
            selected_documents = st.multiselect("Documents to use", document_options, default=document_options, key="chair_pdf_documents")
            page_ranges = {}
            for position, document in enumerate(selected_documents):
                maximum = int(pdf_pages.loc[pdf_pages.document == document, "page"].max())
                if maximum == 1:
                    selected_range = (1, 1)
                    st.caption(f"{document}: page 1")
                else:
                    selected_range = st.slider(f"{document}: page range", 1, maximum, (1, maximum), key=f"chair_pdf_range_{position}")
                page_ranges[document] = selected_range
            pdf_keywords = st.text_area("Keywords or phrases (optional; comma, semicolon or new line separated)", placeholder="absorption; regional development; renewable energy", key="chair_pdf_keywords")
            selected_evidence = select_pdf_evidence(pdf_pages, selected_documents, page_ranges, pdf_keywords)
            e1, e2, e3 = st.columns(3)
            e1.metric("Selected documents", len(selected_documents))
            e2.metric("Selected pages", len(selected_evidence))
            e3.metric("Extracted characters", f"{int(selected_evidence.characters.sum()) if not selected_evidence.empty else 0:,}")
            if selected_evidence.empty:
                st.warning("No PDF pages match the current document/page/keyword scope.")
            else:
                guided_dataframe("Selected PDF evidence", selected_evidence[["document", "page", "characters", "text"]], max_rows=100)
                st.download_button("Download selected PDF evidence index", selected_evidence.to_csv(index=False).encode("utf-8-sig"), "selected_pdf_evidence.csv", "text/csv")

    with prompt_tab:
        st.markdown("#### Copy-ready questions and analytical commands")
        st.info("Replace square-bracket placeholders before use. Prompts marked Research Chair can be pasted into the question box. Prompts marked Module describe the exact analysis to configure in the named specialist module; unsupported estimators are never silently imitated.")
        prompts = prompt_library()
        p1, p2 = st.columns(2)
        with p1:
            prompt_category = st.selectbox("Prompt category", ["All"] + list(prompts.category.drop_duplicates()), key="prompt_library_category")
        with p2:
            prompt_module = st.selectbox("Dashboard module", ["All"] + list(prompts.module.drop_duplicates()), key="prompt_library_module")
        prompt_search = st.text_input("Search prompts", placeholder="Example: Monte Carlo, limitations, clustering, equation", key="prompt_library_search")
        shown_prompts = prompts.copy()
        if prompt_category != "All":
            shown_prompts = shown_prompts[shown_prompts.category == prompt_category]
        if prompt_module != "All":
            shown_prompts = shown_prompts[shown_prompts.module == prompt_module]
        if prompt_search.strip():
            pattern = re.escape(prompt_search.strip())
            keep = shown_prompts.astype(str).apply(lambda column: column.str.contains(pattern, case=False, regex=True)).any(axis=1)
            shown_prompts = shown_prompts[keep]
        st.caption(f"{len(shown_prompts):,} copy-ready prompt(s) match the current filters.")
        for number, row in enumerate(shown_prompts.itertuples(index=False), start=1):
            with st.expander(f"{number}. {row.category} · {row.module} · {row.where_to_use}", expanded=number <= 2):
                st.code(row.copy_ready_prompt, language=None)
                st.markdown(f"**Required setup:** {row.required_setup}\n\n**Expected output:** {row.expected_output}")
        pd1, pd2 = st.columns(2)
        with pd1:
            st.download_button("Download Prompt Library (CSV)", prompts.to_csv(index=False).encode("utf-8-sig"), "makryvelios_prompt_library.csv", "text/csv")
        with pd2:
            st.download_button("Download Prompt Library (Markdown)", prompt_library_markdown(prompts).encode("utf-8"), "makryvelios_prompt_library.md", "text/markdown")

    protocol = {
        "working_title": working_title,
        "research_question": research_question,
        "algorithm": algorithm,
        "outcome": outcome,
        "predictors": predictors,
        "year_column": chair_year,
        "year_range": [chair_start, chair_end],
        "group_column": group_column,
        "aggregation": aggregation,
        "equation": equation,
        "derived_variable": derived_name,
        "derived_expression": derived_expression,
        "derived_role": derived_role,
        "steps": steps,
        "limitations": limitations,
        "researcher_notes": researcher_notes,
        "selected_columns": chair_columns,
        "selected_pdf_documents": selected_documents,
        "pdf_keywords": pdf_keywords,
    }

    with quick_tab:
        st.markdown("#### One-screen guided chat autopilot")
        st.success("Use this page if you do not want to configure the original analytical menu. The Chair uses the active spreadsheet automatically, lets you run one or many editable questions, and returns computed tables, genuine prose answers, figures and a paper-ready bundle.")
        st.markdown("**Three actions only:** 1) choose questions; 2) optionally map model variables; 3) press **Run selected questions**.")

        all_prompts = prompt_library().reset_index(drop=True)
        prompt_labels = {
            f"{index + 1:02d} · {row.category} · {row.copy_ready_prompt[:105]}": row.copy_ready_prompt
            for index, row in all_prompts.iterrows()
        }
        core_phrases = [
            "five most important facts", "most striking statistical result", "greatest missingness",
            "safely conclude", "not conclude", "limitations section", "additional analyses",
            "section-by-section outline of the methodology", "section-by-section outline of the results",
            "research questions and concise evidence-based answers",
        ]
        core_labels = [
            label for label, prompt in prompt_labels.items()
            if any(phrase in prompt.casefold() for phrase in core_phrases)
        ]
        selected_prompt_labels = st.multiselect(
            "Questions to run",
            list(prompt_labels),
            default=core_labels,
            key="autopilot_prompt_selection",
            help="Choose a short core report, any individual questions, or all available questions. Every selected question remains editable before execution.",
        )
        preset1, preset2, preset3 = st.columns(3)
        with preset1:
            load_core = st.button("Load core paper questions", key="auto_load_core")
        with preset2:
            load_selected = st.button("Load current selection", key="auto_load_selected")
        with preset3:
            load_all = st.button("Load all possible questions", key="auto_load_all")
        if "autopilot_question_editor" not in st.session_state:
            st.session_state["autopilot_question_editor"] = "\n".join(prompt_labels[label] for label in core_labels)
        if load_core:
            st.session_state["autopilot_question_editor"] = "\n".join(prompt_labels[label] for label in core_labels)
        if load_selected:
            st.session_state["autopilot_question_editor"] = "\n".join(prompt_labels[label] for label in selected_prompt_labels)
        if load_all:
            st.session_state["autopilot_question_editor"] = "\n".join(all_prompts.copy_ready_prompt.astype(str))
        edited_questions = st.text_area(
            "Editable questions — one question per line",
            key="autopilot_question_editor",
            height=280,
            help="Change any wording, add a new question on a new line, or delete questions you do not want.",
        )
        questions_to_run = [line.strip() for line in edited_questions.splitlines() if line.strip()]
        st.caption(f"{len(questions_to_run):,} question(s) will run. Prepared text is shown on a white background with black type.")

        use_entire_dataset = st.checkbox(
            "Use every row and column in the active spreadsheet",
            value=True,
            key="autopilot_all_data",
            help="Recommended for first use. Untick only when you deliberately configured the Advanced evidence scope.",
        )
        auto_scope = df.copy() if use_entire_dataset else scoped_preview.copy()
        auto_numeric = list(auto_scope.select_dtypes(include=np.number).columns)
        auto_non_numeric = [c for c in auto_scope.columns if c not in auto_numeric]
        auto_named_years = [c for c in auto_scope.columns if any(token in str(c).casefold() for token in ["year", "date", "έτος", "ετος"])]
        with st.expander("Quick model settings — needed only for regression, Monte Carlo, prediction, group or longitudinal models", expanded=False):
            st.caption("General questions—headline results, missingness, correlations, conclusions, limitations and paper outlines—run without these settings. The Chair never guesses a dependent variable.")
            auto_outcome = st.selectbox("Outcome / measure to explain", [None] + auto_numeric, key="auto_outcome")
            auto_predictors = st.multiselect("Predictors / explanatory measures", [c for c in auto_numeric if c != auto_outcome], key="auto_predictors", max_selections=200)
            auto_year = st.selectbox("Year/time variable", [None] + auto_named_years + [c for c in auto_scope.columns if c not in auto_named_years], key="auto_year")
            auto_group = st.selectbox("Group/entity variable", [None] + auto_non_numeric + [c for c in auto_scope.columns if c not in auto_non_numeric], key="auto_group")

        auto_protocol = dict(protocol)
        auto_protocol.update({
            "working_title": working_title or "Research Chair analytical report",
            "outcome": auto_outcome,
            "predictors": auto_predictors,
            "year_column": auto_year,
            "group_column": auto_group,
            "selected_columns": list(auto_scope.columns),
            "autopilot": True,
        })
        if st.button(
            "Run selected questions",
            type="primary",
            key="run_autopilot_questions",
            disabled=not questions_to_run or (auto_scope.empty and selected_evidence.empty),
        ):
            try:
                baseline = execute_protocol(auto_scope, "Descriptive profile", auto_outcome, auto_predictors, auto_year, auto_group, "Mean", equation)
                auto_result, answer_table, answer_markdown = execute_question_batch(
                    auto_scope, questions_to_run, auto_protocol, baseline, selected_evidence,
                )
                st.session_state["autopilot_result"] = auto_result
                st.session_state["autopilot_answers"] = answer_table
                st.session_state["autopilot_answer_markdown"] = answer_markdown
                st.session_state["autopilot_scope"] = auto_scope
                st.session_state["autopilot_protocol"] = auto_protocol
                st.session_state["autopilot_evidence"] = selected_evidence
            except Exception as exc:
                st.error(f"The selected question batch stopped: {exc}")

        auto_result = st.session_state.get("autopilot_result")
        if auto_result is None:
            st.info("Press Run selected questions. The Chair will display a feasibility verdict for every question and will never claim that an unsupported analysis ran.")
        else:
            auto_answers = st.session_state.get("autopilot_answers", pd.DataFrame())
            auto_answer_markdown = st.session_state.get("autopilot_answer_markdown", "")
            auto_saved_scope = st.session_state.get("autopilot_scope", auto_scope)
            auto_saved_protocol = st.session_state.get("autopilot_protocol", auto_protocol)
            auto_saved_evidence = st.session_state.get("autopilot_evidence", selected_evidence)
            executed_count = int(auto_answers.feasibility_verdict.str.startswith("FEASIBLE").sum()) if not auto_answers.empty else 0
            invalid_count = int(auto_answers.feasibility_verdict.str.contains("ERROR").sum()) if not auto_answers.empty else 0
            unsupported_count = int(auto_answers.feasibility_verdict.str.contains("NOT YET").sum()) if not auto_answers.empty else 0
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Questions assessed", len(auto_answers))
            a2.metric("Executed", executed_count)
            a3.metric("Invalid/incomplete", invalid_count)
            a4.metric("Not yet in chat", unsupported_count)
            st.markdown(auto_answer_markdown)
            guided_dataframe("Question-by-question feasibility and answers", auto_answers, max_rows=250)

            figures = chair_interactive_figures(auto_saved_scope, auto_result)
            if figures:
                st.markdown("#### Interactive analytical figures")
                st.caption("Every figure below is also exported as standalone HTML, 600-dpi PNG and vector SVG/PDF in colour and black-and-white.")
                for title, figure, commentary in figures:
                    st.plotly_chart(figure, width="stretch")
                    st.info(commentary)

            auto_blueprint = build_paper_blueprint(auto_saved_protocol, auto_result, auto_saved_evidence, auto_answer_markdown)
            with st.expander("Complete manuscript-ready draft and outline", expanded=True):
                st.markdown(auto_blueprint)
            auto_bundle = research_bundle(
                auto_saved_scope, auto_saved_protocol, auto_result, auto_saved_evidence,
                auto_blueprint, auto_answer_markdown,
            )
            qd1, qd2, qd3 = st.columns(3)
            with qd1:
                st.download_button("Download all genuine answers (Word)", docx_bytes(auto_answer_markdown), "research_chair_question_answers.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with qd2:
                st.download_button("Download paper draft (Word)", docx_bytes(auto_blueprint), "research_chair_paper_draft.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with qd3:
                st.download_button("Download complete analytical bundle", auto_bundle, "research_command_chair_complete_bundle.zip", "application/zip")

    with answer_tab:
        st.markdown("#### Execute the command and interrogate the evidence")
        if st.button("Run Research Command", type="primary", key="run_research_chair", disabled=scoped_preview.empty and pdf_pages.empty):
            try:
                analytical_scope = scoped_preview.copy()
                executed_expression = ""
                if derived_name.strip() or derived_expression.strip():
                    if not (derived_name.strip() and derived_expression.strip()):
                        raise ValueError("Provide both a derived-variable name and a computable expression, or leave both blank.")
                    analytical_scope = add_safe_derived_column(analytical_scope, derived_name.strip(), derived_expression.strip())
                    executed_expression = f"{derived_name.strip()} = {derived_expression.strip()}"
                effective_outcome = outcome
                effective_predictors = list(predictors)
                if derived_name.strip() and derived_name.strip() in analytical_scope:
                    if derived_role == "Use as outcome":
                        effective_outcome = derived_name.strip()
                    elif derived_role == "Add as predictor" and derived_name.strip() not in effective_predictors:
                        effective_predictors.append(derived_name.strip())
                result = execute_protocol(
                    analytical_scope, algorithm, effective_outcome, effective_predictors, chair_year,
                    group_column, aggregation, equation, executed_expression,
                )
                executed_protocol = dict(protocol)
                executed_protocol["outcome"] = effective_outcome
                executed_protocol["predictors"] = effective_predictors
                st.session_state["chair_scope"] = analytical_scope
                st.session_state["chair_result"] = result
                st.session_state["chair_protocol"] = executed_protocol
                st.session_state["chair_evidence"] = selected_evidence
                st.session_state["chair_messages"] = []
            except Exception as exc:
                st.error(f"Research command failed: {exc}")

        chair_result = st.session_state.get("chair_result")
        if chair_result is None:
            st.info("Define the scope and protocol, then press Run Research Command. Nothing is inferred until the command is executed.")
        else:
            saved_scope = st.session_state.get("chair_scope", pd.DataFrame())
            saved_protocol = st.session_state.get("chair_protocol", protocol)
            saved_evidence = st.session_state.get("chair_evidence", pd.DataFrame())
            st.success(f"Protocol executed: {chair_result.algorithm}. The saved analytical scope contains {len(saved_scope):,} records.")
            for table_name, result_table in chair_result.tables.items():
                table_with_downloads(table_name, result_table, f"chair_{table_name.lower().replace(' ', '_')}", max_rows=50_000)
            trend = chair_result.tables.get("Longitudinal results", pd.DataFrame())
            if not trend.empty and "year" in trend:
                value_columns = [c for c in trend.select_dtypes(include=np.number).columns if c != "year"]
                if value_columns:
                    long_trend = trend.melt(id_vars=[c for c in trend.columns if c not in value_columns], value_vars=value_columns, var_name="measure", value_name="value")
                    colour = saved_protocol.get("group_column") if saved_protocol.get("group_column") in long_trend else "measure"
                    trend_fig = px.line(long_trend, x="year", y="value", color=colour, line_dash="measure" if colour != "measure" else None, markers=True, title="Selected longitudinal evidence")
                    figure_with_downloads(trend_fig, "research_chair_longitudinal", long_trend, "The figure reports only the selected years, rows, variables and aggregation. Apparent trends require formal temporal and causal evaluation before policy interpretation.")
            monte_carlo_draws = chair_result.tables.get("Monte Carlo draws", pd.DataFrame())
            if not monte_carlo_draws.empty:
                monte_terms = [c for c in monte_carlo_draws.columns if c not in ["simulation", "const"]]
                if monte_terms:
                    selected_term = monte_terms[0]
                    monte_fig = px.histogram(
                        monte_carlo_draws,
                        x=selected_term,
                        nbins=60,
                        marginal="box",
                        title=f"Monte Carlo coefficient distribution: {selected_term}",
                    )
                    monte_fig.add_vline(x=0, line_dash="dash", line_color="#FFFFFF")
                    figure_with_downloads(
                        monte_fig,
                        "research_chair_monte_carlo",
                        monte_carlo_draws[["simulation", selected_term]],
                        "The distribution shows how the selected coefficient changes across reproducible simulated samples. Values concentrated on one side of zero indicate sign stability under the stated simulation mechanism; they do not establish causality.",
                    )
            st.markdown("##### Methodological interpretation")
            for comment in chair_result.comments:
                st.info(comment)

            local_models = ollama_models()
            external_llm = st.session_state.get("llm_config", {})
            engine_options = ["Built-in offline interpreter"] + (["Local Ollama"] if local_models else []) + (["External LLM API"] if llm_configured(external_llm) else [])
            engine = st.radio("Natural-language engine", engine_options, horizontal=True, key="chair_engine")
            if not local_models:
                st.caption("Optional enhancement: install Ollama locally and download a model. Streamlit Cloud will continue using the free built-in interpreter.")
                selected_model = None
            else:
                selected_model = st.selectbox("Local Ollama model", local_models, key="chair_ollama_model")

            with st.expander("Example questions and the standard of answer to expect", expanded=False):
                st.markdown("""
**Questions understood by the free analytical interpreter**

- `What is the most striking statistical result?`
- `Which variable has the greatest missingness, and can it be modelled safely?`
- `Which two indicators have the strongest association?`
- `What changed most through the selected years?`
- `Run OLS for the selected outcome and predictors and explain the coefficients.`
- `Which methods are explicitly mentioned in the selected PDF pages?`
- `Run the analysis as in the paper and prepare the result tables.`

**Illustrative answer structure — “What is the most striking statistical result?”**

> **Direct answer:** identifies the leading pair and reports Spearman ρ, sample size and p-value.  
> **Why it matters:** explains whether the magnitude is scientifically unusual or suggests overlapping measures.  
> **Data-quality warning:** reports the worst missing variable and its usable observation count.  
> **Model warning:** rejects identifier codes or year fields as substantively meaningless regression variables.  
> **Publication-ready wording:** supplies a cautious paragraph based on the computed result.  
> **Next analysis:** recommends the exact diagnostic or model that should follow.

The numerical answer is recalculated from the currently saved rows and variables; the example is not a fixed response.
""")

            for message in st.session_state.get("chair_messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            natural_question = st.text_area("Ask about the selected data, years, equations, results, PDFs or paper structure", placeholder="What can I conclude, what can I not conclude, and how should this be written in the Results section?", key="chair_natural_question")
            ask_col, clear_col = st.columns([3, 1])
            with ask_col:
                ask_chair = st.button("Ask the Research Chair", disabled=not natural_question.strip(), key="ask_research_chair")
            with clear_col:
                clear_chair = st.button("Clear conversation", key="clear_research_chair")
            if clear_chair:
                st.session_state["chair_messages"] = []
                st.rerun()
            if ask_chair:
                try:
                    computed_result, deterministic_reply = execute_natural_language_command(
                        saved_scope, natural_question, saved_protocol, chair_result, saved_evidence,
                    )
                    if engine == "Local Ollama" and selected_model:
                        reply = ollama_reply(natural_question, saved_protocol, computed_result, saved_evidence, selected_model)
                    elif engine == "External LLM API":
                        evidence_payload = {
                            "question": natural_question,
                            "algorithm": getattr(computed_result, "algorithm", "Research Chair"),
                            "protocol": saved_protocol,
                            "computed_comments": list(getattr(computed_result, "comments", [])),
                            "computed_tables": {name: table.head(120).to_dict("records") for name, table in getattr(computed_result, "tables", {}).items()},
                            "pdf_evidence": saved_evidence.head(40).to_dict("records") if isinstance(saved_evidence, pd.DataFrame) else [],
                            "deterministic_interpretation": deterministic_reply,
                        }
                        reply = llm_reply(
                            "Answer the research question using ONLY the computed evidence below. Preserve all numerical values, distinguish computation from interpretation, identify limitations, and provide publication-ready wording where appropriate.\n\n" + json.dumps(evidence_payload, ensure_ascii=False, default=str),
                            external_llm,
                        )
                    else:
                        reply = deterministic_reply
                    st.session_state["chair_result"] = computed_result
                    messages = st.session_state.setdefault("chair_messages", [])
                    if len(messages) >= 2 and messages[-2].get("role") == "user" and messages[-2].get("content", "").strip() == natural_question.strip():
                        del messages[-2:]
                    messages.extend([
                        {"role": "user", "content": natural_question},
                        {"role": "assistant", "content": reply},
                    ])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Natural-language response failed: {exc}")

            latest_reply = next((m["content"] for m in reversed(st.session_state.get("chair_messages", [])) if m["role"] == "assistant"), "")
            blueprint = build_paper_blueprint(saved_protocol, chair_result, saved_evidence, latest_reply)
            with st.expander("Paper-writing blueprint preview", expanded=True):
                st.markdown(blueprint)
            bundle = research_bundle(saved_scope, saved_protocol, chair_result, saved_evidence, blueprint, latest_reply)
            b1, b2, b3 = st.columns(3)
            with b1:
                st.download_button("Download paper blueprint (Word)", docx_bytes(blueprint), "research_paper_blueprint.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with b2:
                st.download_button("Download paper blueprint (Markdown)", blueprint.encode("utf-8"), "research_paper_blueprint.md", "text/markdown")
            with b3:
                st.download_button("Download complete Research Chair bundle", bundle, "research_command_chair_bundle.zip", "application/zip")
            st.warning("Scientific safeguard: PDF passages are notes, not automatically verified citations; equations are documented instructions; custom algorithms are not treated as validated estimators unless implemented and tested. Review every claim before submission.")


elif page == "13. Methods & reproducibility":
    module_guide("Document the analytical coverage, implementation choices and reproducibility route.", "Use the catalogue to identify an estimator, then retain the exported data, model configuration, seed and software version with the manuscript.", "Reproducibility requires the exact data version and transformations, not only the estimator name.")
    st.subheader("Implemented analytical families")
    methods = {
        "Data engineering": "Simultaneous CSV/XLS/XLSX/TSV ingestion; all-sheet reading; separate analysis, side-by-side column alignment, row append, keyed join and source lineage; embedded-header repair; data dictionary; missingness and duplicate checks.",
        "Descriptive": "Mean, standard error, dispersion, quantiles, skewness, kurtosis, CV, frequency tables; Pearson/Spearman/Kendall correlations with p-values.",
        "Tests": "Welch t, Mann–Whitney, one-way ANOVA, Kruskal–Wallis, Levene, chi-square/Cramér's V, Shapiro–Wilk, D'Agostino K² and Anderson–Darling.",
        "Econometrics": "OLS/WLS, Huber robust regression; HC0–HC3, HAC and clustered covariance; logit, probit, Poisson, negative-binomial, fractional logit, Gamma log-link and quantile regression; fixed effects; IV/2SLS; difference-in-differences; Ridge/Lasso/Elastic Net; VIF, BP/White, RESET, DW, JB and Cook's D.",
        "High-dimensional": "Vectorised 1,000-outcome × 1,000-predictor OLS screening; SVD/pseudoinverse; Benjamini–Hochberg and Bonferroni corrections.",
        "Panel / R&D": "Start-year, end-year or active-year allocation; project and region–year EE1–EE9 specifications; pooled OLS, two-way fixed effects, random effects, entity-clustered covariance and Hausman specification testing.",
        "Spatial": "Official 1:1m GISCO NUTS-2/NUTS-3 boundaries; bilingual aliases; choropleths; KNN Moran's I; local Moran/LISA proxy; 999 permutations; colour and monochrome exports.",
        "Multivariate/time": "PCA, legacy k-means, Cronbach's alpha/item diagnostics, ADF/KPSS stationarity, Granger predictability and ARIMA forecasts.",
        "Advanced clustering": "One-dimensional or multivariate K-means, hierarchical agglomerative, Gaussian mixture and DBSCAN; standard/robust/min-max scaling; automatic k; silhouette, Calinski–Harabasz, Davies–Bouldin, profiles and PCA projection.",
        "Predictive validation": "Cross-validated OLS, Ridge, Lasso, Elastic Net, random forest, extra trees and gradient boosting; out-of-fold RMSE/MAE/R², predictions and permutation importance.",
        "Decision support": "Econometric shock approximation and constrained linear-programming allocation.",
        "Multi-criteria decision analysis": "MAVT, TOPSIS and PROMETHEE II rankings; equal, user-defined, Entropy, CRITIC and AHP pairwise weights; AHP consistency; one-at-a-time weight sensitivity; Monte Carlo rank acceptability and method-agreement diagnostics.",
        "Iterative Trichotomic Approach": "Eligibility gates; C1-C6 weighted scoring; ranking by call; exact binary portfolio optimisation under call envelopes and beneficiary caps; ITA-PB policy/equity rounds; published converging weights; Hybrid ITA-RW score-and-weight uncertainty; project inclusion probabilities; green/gray/red freezing; robustness index; deterministic scorecards; regional allocation; complete GAMS-ready export.",
        "GAMS-compatible ITA Studio": "Visible GAMS-style sets, parameters, X(p), PORTFSCORE and equations; original SYN2 540 and R&D 2437 presets; regional/sector/intervention ceilings; hard GREEN/RED fixing; effective-budget rules; HiGHS execution; solver/equation/variable diagnostics; scenario-overlap outputs; Monte Carlo; API-key-free high-detail Greece maps with bundled NUTS-2/NUTS-3 boundaries and 600-dpi/vector export; original-source viewer; generated .gms/.prn/.lst-style packages; optional user-key LLM interpretation after computation.",
        "Expert respondent analytics": "Respondent-level data-quality checks; raw and normalised preference distributions; bootstrap mean intervals; Kendall concordance; Spearman dependence; PCA; automatically selected K-means preference segments; subgroup Kruskal-Wallis tests with Benjamini-Hochberg correction and epsilon-squared effects; empirical weight-scenario bridge to Hybrid ITA-RW.",
        "Monte Carlo": "Wild/residual/parametric OLS simulation with full coefficient draws, bias, Monte Carlo standard errors and percentile intervals; stochastic cost-benefit R&D portfolio selection with selection probabilities and downside distributions.",
        "Research Command Chair": "Free/offline one-screen question-batch autopilot plus retained advanced XLSX/PDF protocol builder; editable selected/all-question sets; feasibility verdicts; exact evidence scoping; safe equations; descriptive, correlation, OLS, Monte Carlo, normality, outlier, group-test, PCA, reliability, clustering, prediction, ARIMA and panel execution when required variable roles are mapped; genuine paper prose and complete analytical bundles.",
        "Frontier methods": "Multi-objective Pareto and robust binary portfolio optimisation; cross-fitted doubly robust AIPW causal estimation with overlap/balance diagnostics; offline Bayesian posterior and posterior-predictive regression; SHAP TreeExplainer with deterministic fallback; DuckDB read-only acceleration and Arrow/Parquet interchange.",
        "Agentic Research Mode": "Standalone approval-gated research runner that works without AI but can now use Gemini, Groq, Claude, Ollama or a compatible model for a second-pass evidence synthesis. It indexes PDF evidence by page, generates up to 150 research questions, executes bounded deterministic analyses, maintains multi-turn research conversation, and can rewrite Abstract/Results/Discussion/Conclusion/run-specific Limitations from retrieved computed rows without changing any numerical result. The submission bundle includes DOCX, XLSX, JSON, HTML, 600-dpi/vector graphics and reproducibility manifest.",
        "Outputs": "Exact tables, XLSX/CSV/Parquet, self-contained HTML/JavaScript report, 600-dpi PNG and vector SVG/PDF figures in colour and black-and-white, JSON manifests and near-submission DOCX bundles.",
    }
    guided_dataframe("Methods and capabilities", pd.DataFrame(methods.items(), columns=["family", "capabilities"]))
    st.subheader("Source-to-method evidence catalogue")
    guided_dataframe("Source-to-method evidence catalogue", EVIDENCE)
    st.download_button("Download source-to-method catalogue", to_excel_bytes({"Evidence catalogue": EVIDENCE, "Methods": pd.DataFrame(methods.items(), columns=["family", "capabilities"])}), "methods_and_source_evidence.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Optional R replication")
    r_available = shutil.which("Rscript") is not None
    st.write("Rscript detected." if r_available else "Rscript is not installed on this machine. Python functionality is unaffected; deploy R only if independent replication is required.")
    st.code("streamlit run app.py", language="bash")
    st.code("pip install -r requirements.txt", language="bash")
    st.caption("The app never sends raw uploaded datasets to a paid/external LLM automatically. Optional Ollama remains local; the user-key LLM co-pilot sends only a compact computed-evidence summary after an explicit LLM action. Eurostat GISCO is contacted solely to retrieve public map boundaries unless a custom GeoJSON is supplied.")

    st.subheader("Version 5.8.4 documentation library")
    st.info("The v5.5.0 technical report remains the architectural reference for the retained workbench. Version 5.8.4 is strictly additive over v5.8.3. It hardens the Agentic full-draft synthesis contract: Groq GPT-OSS models now receive strict JSON-schema output instructions at API level, while the synthesis parser can also recover complete sectioned prose from the same response without changing provider/model. No automatic provider/model fallback is used. Version 5.8.3 hardened research-question generation; v5.8.2 added Gemini/Groq/Ollama and AI synthesis; all earlier analytical, GAMS/ITA, mapping, respondent, Research Chair, Frontier and export capabilities remain available.")
    documentation_files = [
        ("Complete technical documentation — Word", "Makryvelios_Technical_Documentation_v5_2_1.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("Complete technical documentation — PDF", "Makryvelios_Technical_Documentation_v5_2_1.pdf", "application/pdf"),
        ("Complete searchable documentation — Markdown", "COMPLETE_DOCUMENTATION.md", "text/markdown"),
        ("Quick start", "QUICK_START.md", "text/markdown"),
        ("Dedicated MCDA guide", "MCDA_GUIDE.md", "text/markdown"),
        ("Deployment and operations", "DEPLOYMENT_AND_OPERATIONS.md", "text/markdown"),
        ("Validation and QA", "VALIDATION_AND_QA.md", "text/markdown"),
        ("Requirements coverage", "REQUIREMENTS_COVERAGE.md", "text/markdown"),
        ("Research Command Chair guide", "RESEARCH_COMMAND_CHAIR_GUIDE.md", "text/markdown"),
        ("All-dataset column-combination guide", "DATA_COMBINATION_GUIDE_v5_5_1.md", "text/markdown"),
        ("ITA decision-support guide", "ITA_DECISION_SUPPORT_GUIDE_v5_6_0.md", "text/markdown"),
        ("Expert respondent analytics guide", "EXPERT_RESPONDENT_ANALYTICS_GUIDE_v5_6_1.md", "text/markdown"),
        ("GAMS-compatible ITA Studio guide", "GAMS_COMPATIBLE_ITA_STUDIO_v5_7_0.md", "text/markdown"),
        ("User-key LLM co-pilot guide", "LLM_COPILOT_v5_7_0.md", "text/markdown"),
        ("Frontier methods + Agentic mode guide", "FRONTIER_AGENTIC_GUIDE_v5_8_0.md", "text/markdown"),
        ("AI / LLM Research Engine v5.8.4", "AI_RESEARCH_ENGINE_v5_8_4.md", "text/markdown"),
    ]
    available_docs = [(label, BASE / "documentation" / filename, mime) for label, filename, mime in documentation_files if (BASE / "documentation" / filename).exists()]
    for row_start in range(0, len(available_docs), 2):
        cols = st.columns(2)
        for col, (label, path, mime) in zip(cols, available_docs[row_start:row_start + 2]):
            with col:
                st.download_button(label, path.read_bytes(), path.name, mime, key=f"documentation_{path.name}")
