"""Streamlit interface for the ITA public-funding decision-support module."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_core import to_excel_bytes
from ita import (
    BENEFICIARY_CATEGORY_CAPS, CALL_BUDGETS, compare_portfolios,
    ita_export_bundle, normalise_call_code, prepare_ita_projects,
    run_hybrid_ita, run_policy_ita,
)
from mapping import REGIONS, choropleth_figure, fetch_geojson, match_nuts2, moran_diagnostics


ITA_COLOURS = {
    "Green": "#18C98B", "Gray": "#77808F", "Red": "#E14B64",
    "Stable green": "#18C98B", "Stable red": "#E14B64",
    "Score-sensitive": "#33A6B8", "Weight-sensitive": "#A478E8",
    "Score-and-weight-sensitive": "#F2A93B",
    "Policy-robust green": "#18C98B", "Policy-robust red": "#E14B64",
    "Equity-sensitive gain": "#3F8EFC", "Equity-sensitive loss": "#F2A93B",
    "Policy-conflict zone": "#A478E8",
}


def _probability_matrix(history: pd.DataFrame, projects: pd.DataFrame, *, policy: bool = False, limit: int = 500):
    """Create a compact project-by-round outcome matrix for decision review."""
    value = "selected" if policy else "joint_inclusion_probability"
    matrix = history.pivot(index="project_id", columns="round", values=value)
    if policy:
        order = projects.sort_values(
            ["policy_selected", "policy_rounds_selected", "final_score"], ascending=[False, False, False]
        ).project_id
        scale = [[0.0, "#E14B64"], [0.499, "#E14B64"], [0.5, "#18C98B"], [1.0, "#18C98B"]]
        title = "ITA-PB outcome matrix · project decisions across policy rounds"
        colour_title = "Selected"
    else:
        order = projects.sort_values(
            ["hybrid_selected", "decision_round", "final_score"], ascending=[False, True, False]
        ).project_id
        scale = [
            [0.00, "#8A1737"], [0.05, "#E14B64"], [0.49, "#77808F"],
            [0.51, "#77808F"], [0.95, "#18C98B"], [1.00, "#075E45"],
        ]
        title = "Hybrid ITA-RW outcome matrix · inclusion probability by round"
        colour_title = "Probability"
    order = [identifier for identifier in order if identifier in matrix.index][:limit]
    matrix = matrix.reindex(order)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(float), x=[f"Round {int(c)}" for c in matrix.columns],
        y=matrix.index.astype(str), colorscale=scale, zmin=0, zmax=1,
        colorbar={"title": colour_title, "tickformat": ".0%" if not policy else ".0f"},
        hovertemplate="Project %{y}<br>%{x}<br>Value %{z:.1%}<extra></extra>" if not policy
        else "Project %{y}<br>%{x}<br>Selected %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=min(1250, max(520, len(matrix) * 2.2)), margin=dict(l=10, r=20, t=70, b=20))
    fig.update_yaxes(showticklabels=len(matrix) <= 80, autorange="reversed", title=None)
    fig.update_xaxes(side="top", title=None)
    return fig, len(matrix), len(history.project_id.unique())


def _round_flow(history: pd.DataFrame):
    """Visualise movement between trichotomic states over successive rounds."""
    wide = history.pivot(index="project_id", columns="round", values="round_classification")
    rounds = list(wide.columns)
    states = ["Green", "Gray", "Red"]
    labels = [f"R{int(r)} · {state}" for r in rounds for state in states]
    node_index = {(r, state): i for i, (r, state) in enumerate((r, state) for r in rounds for state in states)}
    sources, targets, values, link_colours = [], [], [], []
    rgba = {"Green": "rgba(24,201,139,.50)", "Gray": "rgba(119,128,143,.40)", "Red": "rgba(225,75,100,.50)"}
    for left, right in zip(rounds[:-1], rounds[1:]):
        transitions = wide.groupby([left, right], dropna=False).size()
        for (before, after), count in transitions.items():
            if before in states and after in states and count:
                sources.append(node_index[(left, before)])
                targets.append(node_index[(right, after)])
                values.append(int(count))
                link_colours.append(rgba[str(after)])
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node={"label": labels, "color": [ITA_COLOURS[state] for _ in rounds for state in states], "pad": 18, "thickness": 18},
        link={"source": sources, "target": targets, "value": values, "color": link_colours},
    ))
    fig.update_layout(title="Decision flow across ITA rounds", height=560, margin=dict(l=10, r=10, t=65, b=10))
    return fig


def _uncertainty_scatter(output, green_threshold: float, red_threshold: float):
    """Plot the score-versus-weight uncertainty plane at the last pre-final round."""
    available = sorted(output.inclusion_history["round"].unique())
    chosen_round = available[-2] if len(available) > 1 else available[-1]
    snapshot = output.inclusion_history.loc[output.inclusion_history["round"].eq(chosen_round)].merge(
        output.projects[["project_id", "call", "region", "requested_budget", "uncertainty_zone", "decision"]],
        on="project_id", how="left",
    )
    fig = px.scatter(
        snapshot, x="weight_inclusion_probability", y="score_inclusion_probability",
        color="uncertainty_zone", size="requested_budget", size_max=32,
        color_discrete_map=ITA_COLOURS, hover_name="project_id",
        hover_data={"call": True, "region": True, "requested_budget": ":,.0f", "decision": True},
        labels={
            "weight_inclusion_probability": "Inclusion probability under weight uncertainty",
            "score_inclusion_probability": "Inclusion probability under score uncertainty",
            "uncertainty_zone": "Uncertainty source",
        },
        title=f"Score–weight uncertainty plane · Round {int(chosen_round)}",
    )
    fig.add_vline(x=red_threshold, line_dash="dot", line_color="#E14B64")
    fig.add_vline(x=green_threshold, line_dash="dot", line_color="#18C98B")
    fig.add_hline(y=red_threshold, line_dash="dot", line_color="#E14B64")
    fig.add_hline(y=green_threshold, line_dash="dot", line_color="#18C98B")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"color": "rgba(255,255,255,.25)", "dash": "dash"})
    fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%")
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%")
    fig.update_layout(height=680, legend_title_text="Decision zone")
    return fig


def _funding_utilisation(allocation: pd.DataFrame, call_budgets: dict[str, float]):
    frame = pd.DataFrame({"call": list(call_budgets), "funding_envelope": list(call_budgets.values())})
    frame = frame.merge(allocation[["call", "allocated_budget"]], on="call", how="left").fillna({"allocated_budget": 0.0})
    frame["unallocated_budget"] = (frame.funding_envelope - frame.allocated_budget).clip(lower=0)
    frame = frame.sort_values("funding_envelope", ascending=False)
    fig = go.Figure()
    fig.add_bar(x=frame.call, y=frame.allocated_budget, name="Allocated", marker_color="#18C98B")
    fig.add_bar(x=frame.call, y=frame.unallocated_budget, name="Unallocated", marker_color="#354052")
    fig.update_layout(barmode="stack", title="Funding-envelope utilisation by call", yaxis_title="EUR", height=520)
    return fig, frame


def _regional_spatial_frame(allocation: pd.DataFrame) -> pd.DataFrame:
    if allocation.empty:
        return pd.DataFrame()
    observed = allocation.copy()
    observed["nuts_id"] = match_nuts2(observed["region"])
    observed = observed.dropna(subset=["nuts_id"]).groupby("nuts_id", as_index=False).agg(
        allocated_budget=("allocated_budget", "sum"), selected_projects=("selected_projects", "sum"),
    )
    if observed.empty:
        return observed
    return REGIONS.merge(observed, on="nuts_id", how="left").fillna({"allocated_budget": 0.0, "selected_projects": 0})


def _table(title: str, frame: pd.DataFrame, stem: str, note: str = "", max_rows: int = 10_000) -> None:
    st.subheader(title)
    if note:
        st.caption(note)
    st.dataframe(frame.head(max_rows), width="stretch", hide_index=True)
    st.download_button(
        f"Download {title} (CSV)", frame.to_csv(index=False).encode("utf-8-sig"),
        f"{stem}.csv", "text/csv", key=f"ita_table_{stem}",
    )


def _figure(fig, stem: str, note: str = "") -> None:
    st.plotly_chart(fig, width="stretch")
    if note:
        st.info(note)
    st.download_button(
        "Download interactive figure (HTML)", fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8"),
        f"{stem}.html", "text/html", key=f"ita_figure_{stem}",
    )


def _suggest(tokens: tuple[str, ...], candidates: list[str], excluded: set[str] | None = None):
    excluded = excluded or set()
    return next((column for column in candidates if column not in excluded and any(token in str(column).lower() for token in tokens)), None)


def _required_select(label: str, suggestion, key: str, candidates: list[str]):
    index = candidates.index(suggestion) if suggestion in candidates else 0
    return st.selectbox(label, candidates, index=index, key=key)


def render_ita_module(df: pd.DataFrame) -> None:
    """Render a complete mapping, modelling, review and export workflow."""
    st.subheader("ITA / Public-Funding Decision Support")
    st.markdown(
        '<div class="guide"><b>Purpose.</b> Convert project records into a transparent public-funding portfolio using published ITA logic and the new policy/equity and hybrid extensions.<br>'
        '<b>How to use it.</b> Map project fields and C1-C6, verify funding envelopes, choose uncertainty and policy settings, then run ITA-PB, Hybrid ITA-RW or both.<br>'
        '<b>Interpretation.</b> The module exposes the consequences of scores, weights, uncertainty and policy rules. It supports, but does not replace, accountable administrative judgement.</div>',
        unsafe_allow_html=True,
    )
    st.info("The uploaded data remain unchanged. The model-ready copy records source rows, mapped columns, all assumptions, every round and every project-level decision.")
    all_columns = list(df.columns)
    numeric = list(df.select_dtypes(include=np.number).columns)
    if not all_columns or not numeric:
        st.error("ITA requires project identifiers, grouping fields and numeric budget/criterion columns.")
        return

    st.subheader("1. Project, call and criterion mapping")
    suggested_id = _suggest(("project_id", "project id", "application", "proposal", "κωδ", "έργο", "project"), all_columns)
    suggested_call = _suggest(("invitation", "call", "πρόσκλη", "axis", "at0"), all_columns, {suggested_id})
    suggested_beneficiary = _suggest(("beneficiary", "municipality", "δήμ", "δικαιούχ"), all_columns, {suggested_id, suggested_call})
    suggested_region = _suggest(("region", "περιφέρ", "nuts2"), all_columns)
    suggested_budget = _suggest(("requested budget", "requested_budget", "public expenditure", "budget", "προϋπολογ"), numeric)
    cols = st.columns(5)
    with cols[0]:
        project_id_col = _required_select("Project identifier", suggested_id, "ita_project_id", all_columns)
    with cols[1]:
        call_col = _required_select("Call / invitation", suggested_call, "ita_call", all_columns)
    with cols[2]:
        beneficiary_col = _required_select("Beneficiary / municipality", suggested_beneficiary, "ita_beneficiary", all_columns)
    with cols[3]:
        region_options = [None] + all_columns
        region_col = st.selectbox("Region (optional)", region_options, index=region_options.index(suggested_region) if suggested_region in region_options else 0, key="ita_region")
    with cols[4]:
        budget_col = _required_select("Requested budget", suggested_budget, "ita_budget", numeric)

    criterion_suggestions = [c for c in numeric if re.search(r"(^|[^a-z0-9])c[1-6]([^a-z0-9]|$)", str(c).lower())]
    if len(criterion_suggestions) < 2:
        criterion_suggestions = [c for c in numeric if c != budget_col][:min(6, max(2, len(numeric) - 1))]
    criteria = st.multiselect(
        "Evaluation criteria in C1, C2, ... order", numeric, default=criterion_suggestions[:6], max_selections=12,
        key="ita_criteria", help="C1 must represent developmental need/equity. For Antonis Tritsis, map C1-C6 in the order specified in the methodological document.",
    )
    default_weights = [.25, .20, .20, .15, .15, .05]
    config = pd.DataFrame({
        "criterion": [f"C{i + 1}" for i in range(len(criteria))],
        "source_column": criteria,
        "weight": [default_weights[i] if i < len(default_weights) else 1.0 for i in range(len(criteria))],
    })
    edited_criteria = st.data_editor(
        config, width="stretch", hide_index=True, disabled=["criterion", "source_column"],
        column_config={"weight": st.column_config.NumberColumn("Policy weight", min_value=0.0, format="%.4f")}, key="ita_criterion_config",
    )
    st.caption("Recommended starting weights: C1 25%, C2 20%, C3 20%, C4 15%, C5 15%, C6 5%. Edited weights are normalised and retained verbatim in the reproducibility package.")

    st.subheader("2. Eligibility, geography and observed decision")
    fields = st.columns(4)
    with fields[0]:
        eligibility_columns = st.multiselect("Pass/fail eligibility fields", all_columns, key="ita_eligibility", help="Every selected field must read Yes/True/1/Pass for the project to enter optimisation.")
    with fields[1]:
        disadvantage_col = st.selectbox("Disadvantaged-area flag", [None] + all_columns, key="ita_disadvantaged")
        c1_threshold = st.slider("C1 fallback threshold", 0.0, 10.0, 7.0, .5, key="ita_c1_threshold")
    with fields[2]:
        category_guess = _suggest(("beneficiary category", "category_m", "municipality class", "κατηγορ"), all_columns)
        category_options = [None] + all_columns
        beneficiary_category_col = st.selectbox("Beneficiary category M1-M7", category_options, index=category_options.index(category_guess) if category_guess in category_options else 0, key="ita_beneficiary_category")
    with fields[3]:
        actual_guess = _suggest(("actual selected", "approved", "funded", "status", "ενταγ", "εγκριθ"), all_columns)
        actual_options = [None] + all_columns
        actual_col = st.selectbox("Observed allocation flag", actual_options, index=actual_options.index(actual_guess) if actual_guess in actual_options else 0, key="ita_actual")
        factor_guess = _suggest(("geography cost factor", "cost factor", "island factor", "διορθωτικ"), numeric)
        factor_options = [None] + numeric
        geography_factor_col = st.selectbox("Geography cost factor", factor_options, index=factor_options.index(factor_guess) if factor_guess in factor_options else 0, key="ita_geography_factor")
    scale_scores = st.checkbox("Scale non-standard criterion columns to 0-10", value=False, key="ita_scale_scores", help="0-1 values are multiplied by ten; other out-of-range values use min-max scaling. Leave off for valid C1-C6 scores.")

    st.subheader("3. Funding envelopes and beneficiary caps")
    try:
        observed_calls = sorted({normalise_call_code(value) for value in df[call_col].dropna().unique()})
    except Exception:
        observed_calls = []
    envelope_table = pd.DataFrame({"call": observed_calls, "funding_envelope": [CALL_BUDGETS.get(code, 0.0) for code in observed_calls]})
    edited_envelopes = st.data_editor(
        envelope_table, width="stretch", hide_index=True, disabled=["call"],
        column_config={"funding_envelope": st.column_config.NumberColumn("Funding envelope (EUR)", min_value=0.0, format="€ %.0f")}, key="ita_envelopes",
    )
    cap_table = pd.DataFrame({"category": list(BENEFICIARY_CATEGORY_CAPS), "maximum_funding": list(BENEFICIARY_CATEGORY_CAPS.values())})
    with st.expander("Beneficiary cap schedule M1-M7", expanded=beneficiary_category_col is not None):
        edited_caps = st.data_editor(
            cap_table, width="stretch", hide_index=True, disabled=["category"],
            column_config={"maximum_funding": st.column_config.NumberColumn("Maximum per beneficiary (EUR)", min_value=0.0, format="€ %.0f")}, key="ita_caps",
        )
        if beneficiary_category_col is None:
            st.caption("The M1-M7 schedule is retained in the design but becomes active only after a category column is mapped.")

    st.subheader("4. Iterative, uncertainty and policy design")
    settings_cols = st.columns(4)
    with settings_cols[0]:
        rounds = st.slider("ITA rounds", 2, 6, 4, 1, key="ita_rounds")
        simulations = st.select_slider("Monte Carlo scenarios per round", options=[10, 25, 50, 100, 250, 500, 1000], value=50, key="ita_simulations")
    with settings_cols[1]:
        score_uncertainty = st.slider("Initial score uncertainty (+/- points)", 0.0, 5.0, 1.5, .1, key="ita_score_uncertainty")
        final_budget_factor = st.slider("Final gray-project budget factor", .50, 1.00, .85, .025, key="ita_budget_factor")
    with settings_cols[2]:
        green_threshold = st.slider("Green inclusion threshold", .50, 1.00, .95, .01, key="ita_green_threshold")
        red_threshold = st.slider("Red inclusion threshold", 0.00, .50, .05, .01, key="ita_red_threshold")
    with settings_cols[3]:
        policy_strength = st.slider("C1 policy strength", 0.00, 1.00, .25, .05, key="ita_policy_strength")
        equity_floor = st.slider("Minimum disadvantaged funding share", 0.00, .80, .15, .05, key="ita_equity_floor")
    seed = st.number_input("Reproducible ITA seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="ita_seed")
    st.caption("The new design defaults to 95%/5% green-red thresholds; the earlier EJOR case used 99%/1%. Fifty scenarios are suitable for workflow testing. Publication runs should use substantially more simulations and report convergence.")

    empirical_vectors = st.session_state.get("ita_empirical_weight_vectors")
    empirical_labels = st.session_state.get("ita_empirical_weight_labels", [])
    empirical_source = st.session_state.get("ita_empirical_weight_source", "Expert respondent analytics")
    empirical_available = (
        isinstance(empirical_vectors, np.ndarray) and empirical_vectors.ndim == 2
        and len(empirical_vectors) >= 4 and empirical_vectors.shape[1] == len(criteria)
    )
    weight_sources = ["Published converging-weight schedule"]
    if empirical_available:
        weight_sources.append(f"Empirical respondent distribution ({len(empirical_vectors):,} respondents)")
    weight_source = st.selectbox(
        "Hybrid ITA-RW weight-scenario source", weight_sources, key="ita_weight_scenario_source",
        help="Empirical mode samples complete respondent vectors and progressively contracts them towards their empirical centre across ITA rounds.",
    )
    use_empirical = weight_source.startswith("Empirical")
    if empirical_available:
        st.success(f"Empirical ITA bridge available from {empirical_source}: {len(empirical_vectors):,} respondents × {len(empirical_labels):,} mapped criteria ({', '.join(empirical_labels)}).")
    elif isinstance(empirical_vectors, np.ndarray):
        st.warning(
            f"An empirical respondent matrix is stored, but it has {empirical_vectors.shape[1]} criteria while the project model currently has {len(criteria)}. "
            "Use the same C1-Cn order in both modules before activating it."
        )

    call_budgets = {str(row.call): float(row.funding_envelope) for row in edited_envelopes.itertuples() if pd.notna(row.funding_envelope) and float(row.funding_envelope) > 0}
    category_caps = ({str(row.category): float(row.maximum_funding) for row in edited_caps.itertuples() if pd.notna(row.maximum_funding) and float(row.maximum_funding) > 0} if beneficiary_category_col else {})
    preparation_error = None
    prepared_projects = prepared_criteria = None
    try:
        prepared_projects, prepared_criteria = prepare_ita_projects(
            df, project_id=project_id_col, call=call_col, beneficiary=beneficiary_col, region=region_col,
            requested_budget=budget_col, criteria=list(edited_criteria.source_column),
            weights=pd.to_numeric(edited_criteria.weight, errors="coerce").fillna(0),
            eligibility_columns=eligibility_columns, disadvantaged_column=disadvantage_col,
            beneficiary_category=beneficiary_category_col, actual_selected=actual_col,
            geography_cost_factor=geography_factor_col, scale_scores=scale_scores,
            disadvantaged_c1_threshold=c1_threshold,
        )
        missing_envelopes = sorted(set(prepared_projects.loc[prepared_projects.eligible, "call"]) - set(call_budgets))
        if missing_envelopes:
            preparation_error = "Enter a positive funding envelope for: " + ", ".join(missing_envelopes)
    except Exception as exc:
        preparation_error = str(exc)

    if preparation_error:
        st.error(preparation_error)
    elif prepared_projects is not None:
        metrics = st.columns(4)
        metrics[0].metric("Project records", f"{len(prepared_projects):,}")
        metrics[1].metric("Eligible", f"{int(prepared_projects.eligible.sum()):,}")
        metrics[2].metric("Calls", f"{prepared_projects.call.nunique():,}")
        metrics[3].metric("Funding envelope", f"€{sum(call_budgets.values()):,.0f}")
        if len(prepared_projects) in {2928, 2929}:
            st.warning(f"The input contains {len(prepared_projects):,} records. The methodological note states 2,929, while its call table totals 2,928. The engine uses the uploaded rows and records exact N in every export.")

    disabled = preparation_error is not None or prepared_projects is None or len(criteria) < 2 or red_threshold >= green_threshold
    buttons = st.columns(3)
    with buttons[0]:
        run_policy = st.button("Run ITA-PB", type="primary", disabled=disabled, key="run_ita_policy")
    with buttons[1]:
        run_hybrid = st.button("Run Hybrid ITA-RW", type="primary", disabled=disabled, key="run_ita_hybrid")
    with buttons[2]:
        run_both = st.button("Run both and compare", disabled=disabled, key="run_ita_both")

    if run_policy or run_both:
        try:
            with st.spinner("Solving the four ITA-PB policy rounds..."):
                st.session_state["ita_policy_output"] = run_policy_ita(
                    prepared_projects, call_budgets=call_budgets, beneficiary_category_caps=category_caps,
                    policy_strength=policy_strength, equity_floor=equity_floor,
                )
        except Exception as exc:
            st.error(str(exc))
    if run_hybrid or run_both:
        try:
            with st.spinner("Sampling scores, converging weights and solving repeated binary portfolios..."):
                st.session_state["ita_hybrid_output"] = run_hybrid_ita(
                    prepared_projects, criterion_weights=prepared_criteria.weight, call_budgets=call_budgets,
                    beneficiary_category_caps=category_caps, rounds=rounds, simulations=int(simulations),
                    score_uncertainty=score_uncertainty, final_gray_budget_factor=final_budget_factor,
                    green_threshold=green_threshold, red_threshold=red_threshold, equity_floor=equity_floor, seed=int(seed),
                    empirical_weight_vectors=empirical_vectors if use_empirical else None,
                )
        except Exception as exc:
            st.error(str(exc))

    policy_out = st.session_state.get("ita_policy_output")
    hybrid_out = st.session_state.get("ita_hybrid_output")
    if policy_out is None and hybrid_out is None:
        return
    st.divider()
    st.subheader("ITA decision-support results")
    overview, round_tab, project_tab, geography_tab, robustness_tab, export_tab = st.tabs([
        "Portfolio overview", "Rounds", "Project scorecards", "Municipality / region", "Robustness", "Export & GAMS",
    ])

    with overview:
        active = hybrid_out or policy_out
        summary = active.portfolio_summary.iloc[0]
        metrics = st.columns(4)
        metrics[0].metric("Selected projects", f"{int(summary.selected_projects):,}")
        metrics[1].metric("Allocated funding", f"€{summary.allocated_budget:,.0f}")
        metrics[2].metric("Unallocated envelope", f"€{summary.unallocated_budget:,.0f}")
        metrics[3].metric("Equity index", f"{summary.equity_index:.1%}" if pd.notna(summary.equity_index) else "N/A")
        if policy_out is not None and hybrid_out is not None and prepared_projects is not None:
            comparison = compare_portfolios(prepared_projects, policy_out, hybrid_out)
            _table("Observed, conventional and ITA portfolio comparison", comparison, "ita_portfolio_comparison")
            fig = px.bar(comparison, x="portfolio", y="allocated_budget", color="equity_index", text="selected_projects", title="Portfolio comparison: allocation, project count and equity")
            _figure(fig, "ita_portfolio_comparison", "Differences reflect activated policy, uncertainty and budget-adjustment assumptions; they are not causal effects.")
        if policy_out is not None:
            _table("ITA-PB allocation by call", policy_out.call_allocation, "ita_pb_call_allocation")
        if hybrid_out is not None:
            _table("Hybrid ITA-RW allocation by call", hybrid_out.call_allocation, "hybrid_call_allocation")
        utilisation_fig, utilisation_data = _funding_utilisation(active.call_allocation, call_budgets)
        _figure(
            utilisation_fig, "ita_funding_utilisation",
            "Each column separates the selected portfolio from the still-unallocated part of the corresponding call envelope.",
        )
        st.download_button(
            "Download funding-utilisation data", utilisation_data.to_csv(index=False).encode("utf-8-sig"),
            "ita_funding_utilisation.csv", "text/csv", key="ita_utilisation_data",
        )

    with round_tab:
        if policy_out is not None:
            _table("ITA-PB policy rounds", policy_out.rounds, "ita_pb_rounds")
            fig = px.line(policy_out.rounds, x="round", y="allocated_budget", markers=True, color="equity_index", title="ITA-PB allocation and equity by round")
            _figure(fig, "ita_pb_round_profile", "Round 1 is pure-score optimisation; Round 4 activates C1 priority, the equity floor and beneficiary caps together.")
            matrix_fig, shown, total = _probability_matrix(policy_out.inclusion_history, policy_out.projects, policy=True)
            _figure(
                matrix_fig, "ita_pb_round_matrix",
                f"The matrix shows {shown:,} of {total:,} projects, sorted by final selection and score. The full project-by-round decisions remain downloadable in the tables and reproducibility package.",
            )
        if hybrid_out is not None:
            _table("Hybrid ITA-RW round diagnostics", hybrid_out.rounds, "hybrid_rounds")
            long = hybrid_out.rounds.melt(id_vars=["round"], value_vars=["new_green", "new_red", "remaining_gray"], var_name="set", value_name="projects")
            fig = px.line(
                long, x="round", y="projects", color="set", markers=True,
                color_discrete_map={"new_green": "#18C98B", "new_red": "#E14B64", "remaining_gray": "#77808F"},
                title="Green/red decisions and shrinking gray set",
            )
            _figure(fig, "hybrid_ita_convergence", "Green/red decisions are frozen. The final converged round resolves any remaining gray projects.")
            matrix_fig, shown, total = _probability_matrix(hybrid_out.inclusion_history, hybrid_out.projects)
            _figure(
                matrix_fig, "hybrid_ita_round_matrix",
                f"The modern outcome matrix shows {shown:,} of {total:,} projects, sorted by final decision and decision round. Red and green identify robust exclusion/inclusion; gray identifies the policy-review zone. Full probabilities are retained for every project.",
            )
            _figure(
                _round_flow(hybrid_out.inclusion_history), "hybrid_ita_round_flow",
                "This infographic shows how projects move between Green, Gray and Red as score uncertainty narrows and weights converge.",
            )
            _table("Published converging-weight schedule", hybrid_out.weights_history, "hybrid_converging_weights")

    with project_tab:
        active = hybrid_out or policy_out
        _table("Project-level ITA decisions", active.projects, "ita_project_decisions")
        ids = list(active.scorecards.project_id)
        if ids:
            chosen = st.selectbox("Inspect one project scorecard", ids, key="ita_scorecard_project")
            card = active.scorecards.loc[active.scorecards.project_id.eq(chosen)].iloc[0]
            st.markdown(f"#### {card.project_id} · {card.classification}")
            st.write(card.explanation)
        st.download_button("Download all project scorecards", active.scorecards.to_csv(index=False).encode("utf-8-sig"), "ita_project_scorecards.csv", "text/csv", key="ita_scorecards_download")

    with geography_tab:
        active = hybrid_out or policy_out
        _table("Regional funding profile", active.regional_allocation, "ita_regional_profile")
        _table("Beneficiary / municipality funding profile", active.beneficiary_allocation, "ita_beneficiary_profile")
        if not active.regional_allocation.empty:
            fig = px.bar(active.regional_allocation.head(30).sort_values("allocated_budget"), x="allocated_budget", y="region", orientation="h", color="mean_score", title="Selected funding by region")
            _figure(fig, "ita_regional_allocation", "Population- or need-normalised interpretations require the corresponding socioeconomic denominators.")
        spatial = _regional_spatial_frame(active.regional_allocation)
        if not spatial.empty:
            try:
                geojson = fetch_geojson("NUTS 2 – Regions (13)")
                map_fig = choropleth_figure(spatial, geojson, "allocated_budget")
                map_fig.update_layout(title="ITA selected funding by Greek region")
                _figure(
                    map_fig, "ita_regional_funding_map",
                    "Colour shows selected funding, not effectiveness or need. A per-capita or need-adjusted map requires the corresponding denominator in the uploaded socioeconomic data.",
                )
                global_moran, local_moran = moran_diagnostics(spatial, "allocated_budget", permutations=999, k=3)
                _table("Exploratory spatial autocorrelation", global_moran, "ita_global_moran")
                with st.expander("Local spatial clusters and outliers"):
                    st.dataframe(local_moran, width="stretch", hide_index=True)
                    st.caption("Moran diagnostics use 3-nearest-neighbour weights to accommodate the island geography. Interpret as exploratory spatial evidence, not a causal effect.")
            except Exception as exc:
                st.warning(f"The regional table is available, but the map/spatial diagnostic could not be rendered: {exc}")
        elif region_col:
            st.info("No uploaded region labels matched the 13 bilingual Greek NUTS-2 names/codes. The regional table remains valid; use NUTS codes EL30, EL41-EL43, EL51-EL54 and EL61-EL65 to activate the national map.")
        profile_candidates = [c for c in numeric if c not in set(criteria) | {budget_col}]
        profile_variables = st.multiselect("Additional municipality/profile variables", profile_candidates, default=profile_candidates[:min(6, len(profile_candidates))], key="ita_profile_variables")
        if profile_variables:
            profile = df.groupby(beneficiary_col, dropna=False)[profile_variables].mean(numeric_only=True).reset_index()
            if region_col:
                profile = profile.merge(df[[beneficiary_col, region_col]].drop_duplicates(subset=[beneficiary_col]), on=beneficiary_col, how="left")
            _table("Municipality socioeconomic profile", profile, "ita_municipality_profile")

    with robustness_tab:
        if policy_out is not None:
            _table("ITA-PB policy classification", policy_out.diagnostics, "ita_pb_diagnostics")
            counts = policy_out.projects.policy_classification.value_counts().rename_axis("classification").reset_index(name="projects")
            _figure(px.bar(counts, x="classification", y="projects", color="classification", title="Policy robustness and conflict zones"), "ita_pb_policy_classes", "Equity-sensitive and conflict-zone projects require explicit review, not concealment within a single rank.")
            for comment in policy_out.interpretation:
                st.info(comment)
        if hybrid_out is not None:
            _table("Hybrid uncertainty-source classification", hybrid_out.diagnostics, "hybrid_diagnostics")
            matrix = hybrid_out.projects.groupby(["score_sensitive", "weight_sensitive", "decision"], as_index=False).size().rename(columns={"size": "projects"})
            _table("2x2 score/weight sensitivity matrix", matrix, "hybrid_sensitivity_matrix")
            _figure(
                _uncertainty_scatter(hybrid_out, green_threshold, red_threshold), "hybrid_score_weight_infographic",
                "Each bubble is a project; bubble size is requested funding. Distance from the diagonal reveals whether score or weight uncertainty is the stronger source of instability at the last pre-final round.",
            )
            value = hybrid_out.portfolio_summary.robustness_index.iloc[0]
            st.metric("Portfolio robustness index", f"{value:.1%}" if pd.notna(value) else "N/A")
            for comment in hybrid_out.interpretation:
                st.info(comment)

    with export_tab:
        st.markdown("Each package contains all mapped projects, rounds, inclusion probabilities, converging weights, allocations, scorecards, assumptions and an executable GAMS MIP representation with the project crosswalk.")
        if policy_out is not None:
            st.download_button("Download complete ITA-PB reproducibility + GAMS package", ita_export_bundle(policy_out), "ita_pb_reproducibility_gams.zip", "application/zip", key="ita_policy_bundle")
        if hybrid_out is not None:
            st.download_button("Download complete Hybrid ITA-RW reproducibility + GAMS package", ita_export_bundle(hybrid_out), "hybrid_ita_rw_reproducibility_gams.zip", "application/zip", key="ita_hybrid_bundle")
        if policy_out is not None and hybrid_out is not None and prepared_projects is not None:
            comparison = compare_portfolios(prepared_projects, policy_out, hybrid_out)
            workbook = to_excel_bytes({
                "Portfolio comparison": comparison, "ITA-PB projects": policy_out.projects,
                "ITA-PB rounds": policy_out.rounds, "Hybrid projects": hybrid_out.projects,
                "Hybrid rounds": hybrid_out.rounds, "Hybrid inclusion history": hybrid_out.inclusion_history,
                "Converging weights": hybrid_out.weights_history,
            })
            st.download_button("Download combined ITA evidence workbook", workbook, "ita_complete_evidence.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ita_combined_workbook")
        st.warning("The GAMS file is an independent-replication route. The live dashboard solves the equivalent binary model with SciPy/HiGHS and does not require or claim a GAMS/GUROBI licence.")
