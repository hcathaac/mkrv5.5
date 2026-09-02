"""Streamlit interface for respondent-level expert preference analytics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_core import to_excel_bytes
from respondent import analyse_respondents, respondent_export_bundle


CLUSTER_COLOURS = ["#20D5E6", "#A478E8", "#F2A93B", "#18C98B", "#E14B64", "#4E7CF5"]


def _downloadable_figure(fig, stem: str, note: str = "") -> None:
    st.plotly_chart(fig, width="stretch")
    if note:
        st.caption(note)
    st.download_button(
        "Download interactive figure (HTML)", fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8"),
        f"{stem}.html", "text/html", key=f"respondent_fig_{stem}",
    )


def _table(title: str, frame: pd.DataFrame, stem: str, note: str = "") -> None:
    st.subheader(title)
    if note:
        st.caption(note)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        f"Download {title} (CSV)", frame.to_csv(index=False).encode("utf-8-sig"),
        f"{stem}.csv", "text/csv", key=f"respondent_table_{stem}",
    )


def render_respondent_module(df: pd.DataFrame, source_label: str = "Active dataset") -> None:
    st.subheader("Expert respondent analytics and empirical ITA weights")
    st.markdown(
        '<div class="guide"><b>Purpose.</b> Analyse the complete respondent-level distribution rather than replacing experts with one average weight.<br>'
        '<b>Workflow.</b> Map respondent ID and criterion-preference fields; inspect data quality, distributions, concordance, correlations, preference segments and subgroup heterogeneity; then activate the validated empirical weight matrix for Hybrid ITA-RW.<br>'
        '<b>Boundary.</b> This module is operational now, but the exact confirmatory models and interpretation must be reconciled with the two forthcoming papers before publication.</div>',
        unsafe_allow_html=True,
    )
    numeric = list(df.select_dtypes(include=np.number).columns)
    all_columns = list(df.columns)
    if len(numeric) < 2:
        st.error("The active respondent dataset must contain at least two numeric criterion/preference fields.")
        return

    st.subheader("1. Respondent and preference mapping")
    id_guess = next((c for c in all_columns if any(token in str(c).lower() for token in ("respondent", "expert_id", "response_id", "participant", "id"))), all_columns[0])
    criterion_guess = [c for c in numeric if any(token in str(c).lower() for token in ("weight", "importance", "priority", "c1", "c2", "c3", "c4", "c5", "c6"))]
    if len(criterion_guess) < 2:
        criterion_guess = numeric[:min(6, len(numeric))]
    columns = st.columns(2)
    with columns[0]:
        respondent_id = st.selectbox("Respondent identifier", all_columns, index=all_columns.index(id_guess), key="expert_respondent_id")
        weight_columns = st.multiselect(
            "Criterion preferences in C1, C2, ... order", numeric, default=criterion_guess[:6],
            max_selections=20, key="expert_weight_columns",
            help="Select the respondent-level fields used to derive relative ITA weights. Raw responses remain unchanged and are exported separately.",
        )
    with columns[1]:
        group_options = [None] + [c for c in all_columns if c != respondent_id]
        group_column = st.selectbox("Subgroup / demographic variable (optional)", group_options, key="expert_group_column")
        missing_label = st.selectbox(
            "Missing-response handling", ["Complete respondents only (recommended)", "Criterion-median imputation"],
            key="expert_missing",
        )
        seed = st.number_input("Reproducible analysis seed", 0, 2_147_483_647, 42, 1, key="expert_seed")

    st.info(
        "For the ITA bridge, every valid respondent vector is normalised to sum to one. This converts ratings or allocations into relative criterion weights; it does not assert that the items form a single psychometric scale."
    )
    run = st.button("Run respondent-level analysis", type="primary", disabled=len(weight_columns) < 2, key="run_expert_analysis")
    if run:
        try:
            with st.spinner("Analysing respondent distributions, consensus, heterogeneity and empirical weight scenarios..."):
                output = analyse_respondents(
                    df, respondent_id=respondent_id, weight_columns=weight_columns, group_column=group_column,
                    missing="median" if missing_label.startswith("Criterion") else "complete", seed=int(seed),
                )
                st.session_state["expert_respondent_output"] = output
                st.session_state["expert_respondent_source_label"] = source_label
        except Exception as exc:
            st.error(str(exc))

    output = st.session_state.get("expert_respondent_output")
    if output is None:
        return
    st.divider()
    source = st.session_state.get("expert_respondent_source_label", source_label)
    st.caption(f"Analytical source: {source}")
    summary = output.criterion_summary
    diagnostic_map = output.diagnostics.set_index("check").value.to_dict()
    metrics = st.columns(5)
    metrics[0].metric("Valid respondents", f"{int(diagnostic_map['Valid respondents']):,}")
    metrics[1].metric("Excluded rows", f"{int(diagnostic_map['Excluded rows']):,}")
    metrics[2].metric("Criteria", f"{len(summary):,}")
    metrics[3].metric("Kendall’s W", f"{output.kendall_w:.3f}" if pd.notna(output.kendall_w) else "N/A")
    metrics[4].metric("Preference segments", f"{int(diagnostic_map['Automatically selected clusters']):,}")

    overview, distribution_tab, structure_tab, segments_tab, subgroup_tab, bridge_tab = st.tabs([
        "Summary", "Distributions", "Dependence & consensus", "Preference segments", "Subgroups", "ITA bridge & export",
    ])
    with overview:
        _table(
            "Criterion-level empirical weight summary", summary, "expert_criterion_summary",
            "Confidence intervals are reproducible non-parametric bootstrap intervals for the mean normalised weight.",
        )
        _table("Data-quality and modelling diagnostics", output.diagnostics, "expert_diagnostics")
        strongest = summary.sort_values("mean_weight", ascending=False).iloc[0]
        most_diverse = summary.sort_values("std_dev", ascending=False).iloc[0]
        st.info(
            f"{strongest.criterion} has the largest mean relative weight ({strongest.mean_weight:.1%}). "
            f"{most_diverse.criterion} has the greatest respondent dispersion (SD {most_diverse.std_dev:.3f}). "
            "These are descriptive findings until the papers establish the exact inferential specification."
        )

    with distribution_tab:
        long = output.normalised_weights.copy()
        long["respondent_id"] = output.respondents.respondent_id
        long = long.melt(id_vars="respondent_id", var_name="criterion", value_name="weight")
        violin = px.violin(
            long, x="criterion", y="weight", color="criterion", box=True, points="outliers",
            title="Full respondent distribution of relative criterion weights",
            color_discrete_sequence=CLUSTER_COLOURS,
        )
        violin.update_yaxes(tickformat=".0%")
        _downloadable_figure(
            violin, "expert_weight_distributions",
            "Violin width represents density; the embedded box shows median and interquartile range. Outliers are retained, not deleted.",
        )
        matrix = output.normalised_weights.copy()
        matrix.index = output.respondents.respondent_id
        order = output.respondents.sort_values(["preference_cluster", "PC1"]).respondent_id
        matrix = matrix.reindex(order)
        heatmap = go.Figure(go.Heatmap(
            z=matrix.to_numpy(float), x=matrix.columns, y=matrix.index.astype(str),
            colorscale=[[0, "#102131"], [.5, "#20D5E6"], [1, "#F2A93B"]],
            colorbar={"title": "Relative weight", "tickformat": ".0%"},
            hovertemplate="Respondent %{y}<br>%{x}: %{z:.1%}<extra></extra>",
        ))
        heatmap.update_layout(title="Respondent-by-criterion preference matrix", height=min(1100, max(520, len(matrix) * 3)), margin=dict(l=10, r=10, t=65, b=20))
        heatmap.update_yaxes(showticklabels=len(matrix) <= 100, autorange="reversed")
        _downloadable_figure(heatmap, "expert_respondent_matrix", "Rows are ordered by the automatically detected preference segment and PC1 position.")

    with structure_tab:
        corr = output.correlation.set_index("criterion")
        corr_fig = go.Figure(go.Heatmap(
            z=corr.to_numpy(float), x=corr.columns, y=corr.index,
            zmin=-1, zmax=1, colorscale=[[0, "#E14B64"], [.5, "#182E40"], [1, "#18C98B"]],
            text=np.round(corr.to_numpy(float), 2), texttemplate="%{text}",
            colorbar={"title": "Spearman ρ"}, hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        ))
        corr_fig.update_layout(title="Spearman dependence between respondent preferences", height=620)
        _downloadable_figure(corr_fig, "expert_spearman_matrix", "Because each respondent vector sums to one, some negative correlations arise mechanically from compositional closure and require cautious interpretation.")
        _table("PCA explained variance", output.pca_variance, "expert_pca_variance")
        _table("PCA criterion loadings", output.pca_loadings, "expert_pca_loadings")
        if pd.notna(output.kendall_w):
            description = "limited" if output.kendall_w < .3 else "moderate" if output.kendall_w < .7 else "strong"
            st.info(f"Kendall’s W = {output.kendall_w:.3f}, indicating {description} concordance in the experts’ relative criterion ordering. It measures agreement, not correctness.")

    with segments_tab:
        points = output.respondents.copy()
        points["preference_cluster"] = points.preference_cluster.astype(str)
        pca_fig = px.scatter(
            points, x="PC1", y="PC2", color="preference_cluster", hover_name="respondent_id",
            color_discrete_sequence=CLUSTER_COLOURS, title="Respondent preference segments in PCA space",
            labels={"preference_cluster": "Preference segment"},
        )
        _downloadable_figure(
            pca_fig, "expert_preference_segments",
            "Segments are exploratory K-means groups chosen by the highest silhouette score across feasible k=2–6; they are not respondent types or causal classes.",
        )
        profiles = output.cluster_profiles.melt(id_vars=["preference_cluster", "respondents"], var_name="criterion", value_name="mean_weight")
        profile_fig = px.line(
            profiles, x="criterion", y="mean_weight", color="preference_cluster", markers=True,
            color_discrete_sequence=CLUSTER_COLOURS, title="Preference profile of each respondent segment",
        )
        profile_fig.update_yaxes(tickformat=".0%")
        _downloadable_figure(profile_fig, "expert_segment_profiles")
        _table("Preference-segment profiles", output.cluster_profiles, "expert_cluster_profiles")

    with subgroup_tab:
        if output.subgroup_tests.empty:
            st.info("Map a demographic or professional subgroup field and rerun the analysis to obtain multiplicity-adjusted between-group tests.")
        else:
            _table(
                "Respondent subgroup heterogeneity", output.subgroup_tests, "expert_subgroup_tests",
                "Kruskal-Wallis tests use Benjamini-Hochberg multiplicity adjustment. Epsilon-squared reports effect magnitude; significance does not establish causality.",
            )

    with bridge_tab:
        st.markdown(
            "Activate the validated respondent matrix as the empirical weight-scenario source for Hybrid ITA-RW. "
            "At each round, respondent vectors are sampled and progressively contracted towards their empirical centre; the final round converges to that centre."
        )
        active = st.session_state.get("ita_empirical_weight_vectors") is not None
        cols = st.columns(2)
        with cols[0]:
            if st.button("Activate empirical weights for Hybrid ITA-RW", type="primary", key="activate_empirical_weights"):
                st.session_state["ita_empirical_weight_vectors"] = output.normalised_weights.to_numpy(float)
                st.session_state["ita_empirical_weight_labels"] = list(output.normalised_weights.columns)
                st.session_state["ita_empirical_weight_source"] = source
                st.success(f"Activated {len(output.normalised_weights):,} respondent vectors for the next Hybrid ITA-RW run.")
        with cols[1]:
            if st.button("Clear empirical ITA link", disabled=not active, key="clear_empirical_weights"):
                for key in ("ita_empirical_weight_vectors", "ita_empirical_weight_labels", "ita_empirical_weight_source"):
                    st.session_state.pop(key, None)
                st.success("The ITA module has returned to the published converging-weight baseline.")
        st.download_button(
            "Download respondent analysis + empirical ITA bridge", respondent_export_bundle(output),
            "expert_respondent_analysis_ita_bridge.zip", "application/zip", key="respondent_bundle",
        )
        workbook = to_excel_bytes({
            "Criterion summary": output.criterion_summary, "Respondents": output.respondents,
            "Raw values": output.raw_values, "Empirical weights": output.normalised_weights,
            "Spearman correlation": output.correlation, "PCA variance": output.pca_variance,
            "PCA loadings": output.pca_loadings, "Cluster profiles": output.cluster_profiles,
            "Subgroup tests": output.subgroup_tests, "Diagnostics": output.diagnostics,
        })
        st.download_button(
            "Download complete respondent workbook", workbook, "expert_respondent_analysis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="respondent_workbook",
        )
        with st.expander("Exact ITA bridge metadata"):
            st.code(json.dumps(output.settings, indent=2, ensure_ascii=False), language="json")
