"""Streamlit UI for v5.8.0 Frontier Methods Laboratory."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_core import to_excel_bytes
from frontier_methods import (
    arrow_ipc_bytes,
    bayesian_linear_regression,
    causal_aipw,
    duckdb_available,
    duckdb_query,
    explainable_random_forest,
    pareto_portfolio,
    parquet_bytes,
    pyarrow_available,
)


def _download_html(fig, filename: str, key: str):
    st.download_button(
        "Download interactive HTML",
        fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8"),
        filename,
        "text/html",
        key=key,
    )


def render_frontier_methods(df: pd.DataFrame) -> None:
    st.subheader("Frontier Methods Laboratory")
    st.markdown(
        '<div class="guide"><b>Purpose.</b> Additive frontier methods for questions that exceed conventional OLS/MCDA workflows: Pareto and robust portfolio optimisation, cross-fitted causal estimation, Bayesian posterior analysis, SHAP explainability and large-data execution.<br>'
        '<b>How to use it.</b> Choose the specialised tab, map variables explicitly and run only after the displayed assumptions are appropriate.<br>'
        '<b>Safeguard.</b> These modules extend the workbench; they do not replace any existing estimator, ITA/GAMS workflow, Research Chair or export.</div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs([
        "Pareto / robust optimisation",
        "Causal inference",
        "Bayesian modelling",
        "SHAP / explainable ML",
        "DuckDB / Arrow / Parquet",
    ])
    numeric = list(df.select_dtypes(include=np.number).columns)
    all_columns = list(df.columns)

    with tabs[0]:
        st.markdown("### Multi-objective Pareto portfolio laboratory")
        st.caption("Weighted scalarisation is used only to discover non-dominated portfolios. Reported frontier objective totals remain in the original units.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pid = st.selectbox("Project identifier", all_columns, key="frontier_pareto_id")
        with c2:
            objectives = st.multiselect("Objectives to maximise (2–4)", numeric, max_selections=4, key="frontier_pareto_obj")
        with c3:
            cost = st.selectbox("Project cost", numeric, key="frontier_pareto_cost")
        with c4:
            default_budget = float(pd.to_numeric(df[cost], errors="coerce").dropna().sum() * .35) if cost else 0.0
            budget = st.number_input("Portfolio budget", min_value=0.0, value=max(0.0, default_budget), format="%.2f", key="frontier_pareto_budget")

        st.markdown("#### Robustness controls")
        r1, r2, r3 = st.columns(3)
        with r1:
            robust_lambda = st.number_input("Objective uncertainty penalty λ", min_value=0.0, max_value=10.0, value=0.0, step=.1, key="frontier_robust_lambda")
        with r2:
            cost_unc_opts = [None] + [c for c in numeric if c != cost]
            cost_unc = st.selectbox("Cost uncertainty column (optional)", cost_unc_opts, key="frontier_cost_unc")
        with r3:
            cost_risk = st.number_input("Cost risk multiplier", min_value=0.0, max_value=10.0, value=0.0, step=.1, key="frontier_cost_risk")
        uncertainty_map = {}
        if objectives:
            map_frame = pd.DataFrame({"objective": objectives, "uncertainty_column": [""] * len(objectives)})
            edited = st.data_editor(
                map_frame,
                hide_index=True,
                width="stretch",
                disabled=["objective"],
                column_config={"uncertainty_column": st.column_config.SelectboxColumn(options=[""] + numeric, required=False)},
                key="frontier_unc_map",
            )
            uncertainty_map = {str(r.objective): (str(r.uncertainty_column) if str(r.uncertainty_column) else None) for _, r in edited.iterrows()}
        resolution = st.slider("Pareto weight-grid resolution", 3, 15, 8, key="frontier_pareto_resolution")
        if st.button("RUN PARETO / ROBUST OPTIMISATION", type="primary", key="frontier_pareto_run"):
            try:
                result = pareto_portfolio(
                    df, project_id=pid, objectives=objectives, cost_column=cost, budget=float(budget),
                    uncertainty_columns=uncertainty_map, robust_lambda=float(robust_lambda),
                    cost_uncertainty_column=cost_unc, cost_risk_multiplier=float(cost_risk), resolution=int(resolution),
                )
                st.session_state["frontier_pareto_result"] = result
                st.success(f"Pareto scan complete: {len(result.frontier)} non-dominated portfolios retained.")
            except Exception as exc:
                st.error(f"Pareto optimisation failed: {exc}")
        result = st.session_state.get("frontier_pareto_result")
        if result is not None:
            st.dataframe(result.frontier, width="stretch", hide_index=True)
            if len(result.objective_columns) >= 2:
                x, y = [f"total_{c}" for c in result.objective_columns[:2]]
                fig = px.scatter(result.frontier, x=x, y=y, size="selected_projects", color="effective_cost", hover_name="solution_id", title="Pareto frontier")
                st.plotly_chart(fig, width="stretch")
                _download_html(fig, "pareto_frontier.html", "frontier_pareto_html")
            freq = result.project_frequency.sort_values("pareto_selection_frequency", ascending=False)
            st.markdown("#### Project robustness across Pareto-efficient portfolios")
            st.dataframe(freq, width="stretch", hide_index=True, height=500)
            workbook = to_excel_bytes({"Pareto frontier": result.frontier, "Project frequency": freq})
            st.download_button("Download Pareto workbook", workbook, "pareto_robust_optimisation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="frontier_pareto_xlsx")

    with tabs[1]:
        st.markdown("### Causal inference laboratory")
        st.warning("A causal estimator cannot create identification. Use only with a defensible treatment definition, temporal ordering and pre-treatment adjustment set. The software reports assumptions explicitly and never relabels association as causality automatically.")
        if len(numeric) < 3:
            st.info("At least three numeric variables are needed for the causal laboratory.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                outcome = st.selectbox("Outcome Y", numeric, key="causal_outcome")
            with c2:
                treatment_candidates = [c for c in all_columns if df[c].nunique(dropna=True) == 2]
                treatment = st.selectbox("Binary treatment T", treatment_candidates or all_columns, key="causal_treatment")
            cov_candidates = [c for c in numeric if c != outcome and c != treatment]
            covariates = st.multiselect("Pre-treatment covariates X", cov_candidates, default=cov_candidates[:min(6, len(cov_candidates))], key="causal_covariates")
            folds = st.slider("Cross-fitting folds", 2, 10, 5, key="causal_folds")
            if st.button("ESTIMATE CROSS-FITTED AIPW ATE", type="primary", key="causal_run"):
                try:
                    result = causal_aipw(df, outcome=outcome, treatment=treatment, covariates=covariates, folds=int(folds))
                    st.session_state["causal_result"] = result
                    st.success("Cross-fitted doubly robust estimation complete.")
                except Exception as exc:
                    st.error(f"Causal estimation failed: {exc}")
            result = st.session_state.get("causal_result")
            if result is not None:
                st.dataframe(result.estimate, width="stretch", hide_index=True)
                st.markdown("#### Covariate balance")
                balance_long = result.balance.melt(id_vars="covariate", value_vars=["smd_unweighted", "smd_ipw"], var_name="stage", value_name="SMD")
                fig = px.bar(balance_long, x="SMD", y="covariate", color="stage", barmode="group", orientation="h", title="Standardised mean differences before and after IPW")
                fig.add_vline(x=.1, line_dash="dot"); fig.add_vline(x=-.1, line_dash="dot")
                st.plotly_chart(fig, width="stretch")
                for item in result.assumptions:
                    st.info(item)
                workbook = to_excel_bytes({"ATE": result.estimate, "Balance": result.balance, "Unit diagnostics": result.unit_diagnostics})
                st.download_button("Download causal workbook", workbook, "causal_aipw_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="causal_xlsx")

    with tabs[2]:
        st.markdown("### Bayesian posterior laboratory")
        st.caption("The built-in Gaussian Bayesian regression is fully offline and uses an exact conjugate posterior plus posterior predictive simulation; no API or probabilistic-programming service is required.")
        if len(numeric) >= 2:
            outcome = st.selectbox("Bayesian outcome", numeric, key="bayes_outcome")
            preds = st.multiselect("Bayesian predictors", [c for c in numeric if c != outcome], default=[c for c in numeric if c != outcome][:min(5, len(numeric)-1)], key="bayes_predictors")
            b1, b2, b3 = st.columns(3)
            with b1:
                draws = st.number_input("Posterior draws", 500, 20000, 4000, 500, key="bayes_draws")
            with b2:
                prior_scale = st.number_input("Prior scale (standardised β)", 0.1, 100.0, 10.0, .5, key="bayes_prior")
            with b3:
                seed = st.number_input("Random seed", 0, 2_147_483_647, 580, 1, key="bayes_seed")
            if st.button("RUN BAYESIAN POSTERIOR", type="primary", key="bayes_run"):
                try:
                    result = bayesian_linear_regression(df, outcome=outcome, predictors=preds, draws=int(draws), prior_scale=float(prior_scale), seed=int(seed))
                    st.session_state["bayes_result"] = result
                    st.success("Posterior and posterior predictive distributions generated.")
                except Exception as exc:
                    st.error(f"Bayesian model failed: {exc}")
            result = st.session_state.get("bayes_result")
            if result is not None:
                st.dataframe(result.summary, width="stretch", hide_index=True)
                coef = result.summary[result.summary.term != "Intercept"].copy()
                if not coef.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=coef.posterior_mean, y=coef.term, mode="markers", error_x=dict(type="data", symmetric=False, array=coef["hdi_97.5%"]-coef.posterior_mean, arrayminus=coef.posterior_mean-coef["hdi_2.5%"])))
                    fig.update_layout(title="Posterior coefficient intervals", xaxis_title="Standardised coefficient", height=max(420, 36*len(coef)))
                    st.plotly_chart(fig, width="stretch")
                st.dataframe(result.diagnostics, width="stretch", hide_index=True)
                workbook = to_excel_bytes({"Posterior summary": result.summary, "Posterior draws": result.draws, "Posterior predictive": result.predictive, "Diagnostics": result.diagnostics})
                st.download_button("Download Bayesian workbook", workbook, "bayesian_model_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="bayes_xlsx")

    with tabs[3]:
        st.markdown("### SHAP / explainable machine-learning laboratory")
        st.caption("Numerical prediction remains in scikit-learn. SHAP is used for local/global explanation when available; a deterministic permutation/local-perturbation fallback prevents the module from failing if SHAP is unavailable.")
        if len(numeric) >= 2:
            target = st.selectbox("Target", all_columns, key="xai_target")
            features = st.multiselect("Numeric features", [c for c in numeric if c != target], default=[c for c in numeric if c != target][:min(8, len(numeric))], max_selections=100, key="xai_features")
            local_row = st.number_input("Local explanation row (complete-case index)", 0, max(0, len(df)-1), 0, 1, key="xai_row")
            if st.button("TRAIN + EXPLAIN MODEL", type="primary", key="xai_run"):
                try:
                    result = explainable_random_forest(df, target=target, features=features, local_row=int(local_row))
                    st.session_state["xai_result"] = result
                    st.success(f"Explainability complete using {result.backend}.")
                except Exception as exc:
                    st.error(f"Explainable ML failed: {exc}")
            result = st.session_state.get("xai_result")
            if result is not None:
                st.dataframe(result.performance, width="stretch", hide_index=True)
                fig = px.bar(result.global_importance.head(30).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h", title=f"Global feature importance · {result.backend}")
                st.plotly_chart(fig, width="stretch")
                st.markdown("#### Local explanation")
                st.dataframe(result.local_explanation.head(30), width="stretch", hide_index=True)
                workbook = to_excel_bytes({"Performance": result.performance, "Global importance": result.global_importance, "Local explanation": result.local_explanation, "Predictions": result.predictions})
                st.download_button("Download explainability workbook", workbook, "shap_explainable_ml.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="xai_xlsx")

    with tabs[4]:
        st.markdown("### Large-data execution layer")
        a, b, c = st.columns(3)
        a.metric("Active rows", f"{len(df):,}")
        b.metric("DuckDB", "READY" if duckdb_available() else "INSTALL ON DEPLOY")
        c.metric("Arrow / Parquet", "READY" if pyarrow_available() else "INSTALL ON DEPLOY")
        st.caption("The standard pandas workflow remains intact. DuckDB/Arrow are additive acceleration and interchange layers for larger future datasets.")
        sql = st.text_area("Read-only DuckDB SQL", value="SELECT * FROM active_data LIMIT 100", height=120, key="duckdb_sql")
        if st.button("RUN READ-ONLY SQL", key="duckdb_run"):
            try:
                result = duckdb_query(df, sql)
                st.session_state["duckdb_result"] = result
                st.success(f"Query complete: {len(result):,} rows returned.")
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("duckdb_result")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, width="stretch", hide_index=True)
            st.download_button("Download query CSV", result.to_csv(index=False).encode("utf-8-sig"), "duckdb_query.csv", "text/csv", key="duckdb_csv")
        try:
            pq = parquet_bytes(df)
        except Exception:
            pq = None
        if pq is not None:
            st.download_button("Download active dataset as ZSTD Parquet", pq, "active_dataset.parquet", "application/octet-stream", key="parquet_download")
        try:
            arrow_payload = arrow_ipc_bytes(df)
        except Exception:
            arrow_payload = None
        if arrow_payload is not None:
            st.download_button("Download active dataset as Arrow IPC", arrow_payload, "active_dataset.arrow", "application/vnd.apache.arrow.file", key="arrow_download")
