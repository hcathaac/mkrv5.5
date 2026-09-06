"""Streamlit UI for Makryvelios v5.9 Generalised Confirmatory & Rare Methods."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from analytics_core import to_excel_bytes
from analysis_recipes import RUNNERS, recipe_json, run_recipe
from confirmatory_analytics import (
    AnalysisResult,
    METHOD_CATALOGUE,
    beta_regression,
    conditional_logistic,
    brant_type_wald,
    compositional_transforms,
    cox_proportional_hazards,
    exact_2x2_tests,
    dirichlet_regression,
    dunn_posthoc,
    equivalence_tost,
    firth_logistic,
    gee_regression,
    heckman_two_step,
    latent_class_analysis,
    linear_mixed_effects,
    mantel_haenszel,
    mca_ward,
    meta_analysis,
    page_trend,
    multinomial_logit,
    alexander_govern_test,
    ordered_regression,
    permanova,
    plackett_luce,
    plackett_luce_mixture,
    rasch_1pl,
    regression_discontinuity,
    synthetic_control,
    repeated_rank_tests,
    tobit_regression,
    zero_inflated_count,
    brunner_munzel_test, jonckheere_terpstra, quade_test, cochran_q_test,
    mcnemar_test, bowker_symmetry_test, distance_correlation_test,
    energy_two_sample_test, partial_correlation, meta_regression,
    parsed_numeric_audit,
)


def _run_and_store(key: str, fn, **kwargs) -> None:
    try:
        with st.spinner("Estimating model and building reproducible output..."):
            result = fn(**kwargs)
            st.session_state[key] = result
        st.success("Analysis complete.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def _render_result(result: AnalysisResult | None, key: str) -> None:
    if result is None:
        return
    for warning in result.warnings:
        st.warning(warning)
    if not result.diagnostics.empty:
        st.markdown("#### Diagnostics")
        st.dataframe(result.diagnostics, width="stretch", hide_index=True)
    for title, table in result.tables.items():
        st.markdown(f"#### {title}")
        st.dataframe(table, width="stretch", hide_index=True)
    export = dict(result.tables)
    if not result.diagnostics.empty:
        export["Diagnostics"] = result.diagnostics
    settings = pd.DataFrame({"setting": list(result.settings.keys()), "value": [json.dumps(v, ensure_ascii=False, default=str) for v in result.settings.values()]})
    export["Settings"] = settings
    st.download_button(
        "Download complete analysis workbook",
        to_excel_bytes(export),
        f"{key}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{key}",
    )


def _numeric(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=np.number).columns)


def render_confirmatory_lab(df: pd.DataFrame) -> None:
    st.subheader("Generalised Confirmatory & Rare Methods Laboratory")
    st.markdown(
        '<div class="guide"><b>Purpose.</b> One reusable engine for specialist methods that should not require rebuilding Makryvelios for every paper: separation-safe binary models, ordinal diagnostics, compositional analysis, ranking models, latent categorical structure, clustered/survival models and deliberately rare inferential tools.<br>'
        '<b>Rule.</b> Nothing here is paper-specific. You map variables and select a method family; the same code applies to future datasets.<br>'
        '<b>Safeguard.</b> A method being available does not make it appropriate. Publication-level interpretation still requires the assumptions shown for that method.</div>',
        unsafe_allow_html=True,
    )
    numeric = _numeric(df)
    all_cols = list(df.columns)
    if df.empty:
        st.info("Upload/select a dataset first.")
        return

    tabs = st.tabs([
        "Method catalogue",
        "Binary & ordinal",
        "Bounded & count",
        "Clustered & survival",
        "Compositional",
        "Rankings",
        "Categorical latent structure",
        "Rare utilities",
        "Reusable recipes",
    ])

    with tabs[0]:
        st.caption("The catalogue is intentionally broader than the two current expert-survey papers. Methods are generic and reusable.")
        st.dataframe(METHOD_CATALOGUE, width="stretch", hide_index=True, height=700)
        st.download_button("Download method catalogue", METHOD_CATALOGUE.to_csv(index=False).encode("utf-8-sig"), "confirmatory_method_catalogue.csv", "text/csv", key="confirmatory_catalogue_csv")

    with tabs[1]:
        method = st.selectbox("Binary / ordinal method", ["Firth logistic", "Ordered logit/probit", "Brant-type proportional-odds diagnostic", "Multinomial logit"], key="conf_bo_method")
        if method == "Firth logistic":
            st.caption("Use when ordinary logistic regression is unstable because events are rare or there is complete/quasi-separation.")
            y = st.selectbox("Binary outcome", all_cols, key="firth_y")
            xs = st.multiselect("Predictors", [c for c in all_cols if c != y], key="firth_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="firth_cat")
            if st.button("RUN FIRTH LOGISTIC", type="primary", key="run_firth"):
                _run_and_store("firth_result", firth_logistic, df=df, y=y, x_vars=xs, categorical=cats)
            _render_result(st.session_state.get("firth_result"), "firth_logistic")
        elif method == "Ordered logit/probit":
            y = st.selectbox("Ordinal outcome", all_cols, key="ord_y")
            xs = st.multiselect("Predictors", [c for c in all_cols if c != y], key="ord_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="ord_cat")
            distr = st.selectbox("Link", ["logit", "probit"], key="ord_dist")
            st.caption("For text outcomes, the first-observed category order is used unless the outcome is already numerically coded. Prefer a numeric code or pre-ordered categorical variable for publication work.")
            if st.button("RUN ORDERED MODEL", type="primary", key="run_ord"):
                _run_and_store("ord_result", ordered_regression, df=df, y=y, x_vars=xs, categorical=cats, distribution=distr)
            _render_result(st.session_state.get("ord_result"), "ordered_regression")
        elif method.startswith("Brant"):
            y = st.selectbox("Ordinal outcome", all_cols, key="brant_y")
            xs = st.multiselect("Predictors", [c for c in all_cols if c != y], key="brant_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="brant_cat")
            st.caption("Transparent Brant-type threshold-logit Wald diagnostic. It is explicitly labelled approximate because exact covariance conventions differ between packages.")
            if st.button("RUN BRANT-TYPE DIAGNOSTIC", type="primary", key="run_brant"):
                _run_and_store("brant_result", brant_type_wald, df=df, y=y, x_vars=xs, categorical=cats)
            _render_result(st.session_state.get("brant_result"), "brant_type_wald")
        else:
            y = st.selectbox("Nominal outcome", all_cols, key="multi_y")
            xs = st.multiselect("Predictors", [c for c in all_cols if c != y], key="multi_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="multi_cat")
            if st.button("RUN MULTINOMIAL LOGIT", type="primary", key="run_multi"):
                _run_and_store("multi_result", multinomial_logit, df=df, y=y, x_vars=xs, categorical=cats)
            _render_result(st.session_state.get("multi_result"), "multinomial_logit")

    with tabs[2]:
        method = st.selectbox("Bounded / count method", ["Beta regression", "Tobit censored normal", "Zero-inflated Poisson", "Zero-inflated negative binomial"], key="conf_bc_method")
        if method == "Beta regression":
            y = st.selectbox("Outcome strictly in (0,1)", numeric, key="beta_y")
            xs = st.multiselect("Mean-model predictors", [c for c in all_cols if c != y], key="beta_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="beta_cat")
            prec = st.multiselect("Precision-model numeric predictors (optional)", [c for c in numeric if c != y], key="beta_prec")
            if st.button("RUN BETA REGRESSION", type="primary", key="run_beta"):
                _run_and_store("beta_result", beta_regression, df=df, y=y, x_vars=xs, categorical=cats, precision_vars=prec)
            _render_result(st.session_state.get("beta_result"), "beta_regression")
        elif method.startswith("Tobit"):
            y = st.selectbox("Censored continuous outcome", numeric, key="tobit_y")
            xs = st.multiselect("Predictors", [c for c in all_cols if c != y], key="tobit_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="tobit_cat")
            c1, c2 = st.columns(2)
            with c1:
                use_lower = st.checkbox("Lower censoring bound", value=True, key="tobit_use_lo")
                lower = st.number_input("Lower bound", value=0.0, key="tobit_lo") if use_lower else None
            with c2:
                use_upper = st.checkbox("Upper censoring bound", value=False, key="tobit_use_hi")
                upper = st.number_input("Upper bound", value=1.0, key="tobit_hi") if use_upper else None
            if st.button("RUN TOBIT", type="primary", key="run_tobit"):
                _run_and_store("tobit_result", tobit_regression, df=df, y=y, x_vars=xs, categorical=cats, lower=lower, upper=upper)
            _render_result(st.session_state.get("tobit_result"), "tobit_regression")
        else:
            y = st.selectbox("Count outcome", numeric, key="zi_y")
            xs = st.multiselect("Count-model predictors", [c for c in all_cols if c != y], key="zi_x")
            infl = st.multiselect("Zero-inflation predictors (blank = same as count model)", [c for c in all_cols if c != y], key="zi_infl")
            cats = st.multiselect("Categorical predictors", list(dict.fromkeys(xs + infl)), key="zi_cat")
            model_code = "ZINB" if "negative" in method.lower() else "ZIP"
            if st.button("RUN ZERO-INFLATED MODEL", type="primary", key="run_zi"):
                _run_and_store("zi_result", zero_inflated_count, df=df, y=y, x_vars=xs, inflation_vars=infl, categorical=cats, model=model_code)
            _render_result(st.session_state.get("zi_result"), f"{model_code.lower()}_model")

    with tabs[3]:
        method = st.selectbox("Clustered / survival method", ["Linear mixed effects", "GEE", "Cox proportional hazards"], key="conf_cs_method")
        if method == "Linear mixed effects":
            y = st.selectbox("Continuous outcome", numeric, key="mix_y")
            group = st.selectbox("Grouping / random-intercept unit", all_cols, key="mix_group")
            xs = st.multiselect("Fixed-effect predictors", [c for c in all_cols if c not in {y, group}], key="mix_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="mix_cat")
            rs_opts = [None] + [c for c in numeric if c not in {y}]
            rs = st.selectbox("Random slope (optional)", rs_opts, key="mix_rs")
            if st.button("RUN MIXED MODEL", type="primary", key="run_mix"):
                _run_and_store("mix_result", linear_mixed_effects, df=df, y=y, x_vars=xs, group=group, categorical=cats, random_slope=rs)
            _render_result(st.session_state.get("mix_result"), "linear_mixed_effects")
        elif method == "GEE":
            y = st.selectbox("Outcome", numeric, key="gee_y")
            group = st.selectbox("Cluster / subject identifier", all_cols, key="gee_group")
            xs = st.multiselect("Predictors", [c for c in all_cols if c not in {y, group}], key="gee_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="gee_cat")
            family = st.selectbox("Family", ["Gaussian", "Binomial", "Poisson"], key="gee_family")
            corr = st.selectbox("Working correlation", ["Exchangeable", "Independence"], key="gee_corr")
            if st.button("RUN GEE", type="primary", key="run_gee"):
                _run_and_store("gee_result", gee_regression, df=df, y=y, x_vars=xs, group=group, categorical=cats, family=family, correlation=corr)
            _render_result(st.session_state.get("gee_result"), "gee_regression")
        else:
            time = st.selectbox("Follow-up time", numeric, key="cox_time")
            event = st.selectbox("Event indicator (0/1)", [c for c in numeric if c != time], key="cox_event")
            xs = st.multiselect("Predictors", [c for c in all_cols if c not in {time, event}], key="cox_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="cox_cat")
            strata = st.selectbox("Strata (optional)", [None] + [c for c in all_cols if c not in {time, event}], key="cox_strata")
            if st.button("RUN COX PH", type="primary", key="run_cox"):
                _run_and_store("cox_result", cox_proportional_hazards, df=df, time=time, event=event, x_vars=xs, categorical=cats, strata=strata)
            _render_result(st.session_state.get("cox_result"), "cox_ph")

    with tabs[4]:
        method = st.selectbox("Compositional method", ["CLR / ILR transform", "PERMANOVA", "Dirichlet regression"], key="conf_comp_method")
        comps = st.multiselect("Composition component columns", numeric, key="comp_cols", help="Select parts that jointly represent an allocation/composition, e.g. weights summing to 100.")
        zero = st.number_input("Zero replacement before log-ratios", min_value=1e-12, max_value=0.1, value=1e-6, format="%.8f", key="comp_zero")
        if method == "CLR / ILR transform":
            if st.button("RUN COMPOSITIONAL TRANSFORMS", type="primary", key="run_comp_transform"):
                _run_and_store("comp_transform_result", compositional_transforms, df=df, columns=comps, zero_replacement=float(zero))
            _render_result(st.session_state.get("comp_transform_result"), "compositional_transforms")
        elif method == "PERMANOVA":
            group = st.selectbox("Grouping variable", all_cols, key="perm_group")
            transform = st.selectbox("Geometry", ["ILR (Aitchison)", "CLR", "Raw Euclidean"], key="perm_transform")
            perms = st.number_input("Permutations", min_value=99, max_value=100000, value=999, step=100, key="perm_n")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="perm_seed")
            if st.button("RUN PERMANOVA", type="primary", key="run_permanova"):
                _run_and_store("permanova_result", permanova, df=df, columns=comps, group=group, transform=transform, permutations=int(perms), seed=int(seed), zero_replacement=float(zero))
            _render_result(st.session_state.get("permanova_result"), "permanova")
        else:
            xs = st.multiselect("Predictors", all_cols, key="dir_x")
            cats = st.multiselect("Categorical predictors", [c for c in xs], key="dir_cat")
            if st.button("RUN DIRICHLET REGRESSION", type="primary", key="run_dirichlet"):
                _run_and_store("dir_result", dirichlet_regression, df=df, components=comps, x_vars=xs, categorical=cats, zero_replacement=float(zero))
            _render_result(st.session_state.get("dir_result"), "dirichlet_regression")

    with tabs[5]:
        method = st.selectbox("Ranking method", ["Friedman + Wilcoxon/Holm", "Plackett-Luce", "Plackett-Luce mixture"], key="conf_rank_method")
        cols = st.multiselect("Repeated score / rank columns", numeric, key="rank_cols")
        if method.startswith("Friedman"):
            higher = st.checkbox("Higher value means higher preference", value=True, key="rank_high")
            adjustment = st.selectbox("Multiplicity adjustment", ["holm", "fdr_bh", "bonferroni"], key="rank_adj")
            if st.button("RUN REPEATED-RANK TESTS", type="primary", key="run_rank_tests"):
                _run_and_store("rank_test_result", repeated_rank_tests, df=df, columns=cols, higher_is_better=higher, adjustment=adjustment)
            _render_result(st.session_state.get("rank_test_result"), "repeated_rank_tests")
        elif method == "Plackett-Luce":
            st.caption("Columns must contain within-respondent ranks where smaller rank means earlier/more preferred item.")
            if st.button("RUN PLACKETT-LUCE", type="primary", key="run_pl"):
                _run_and_store("pl_result", plackett_luce, df=df, rank_columns=cols)
            _render_result(st.session_state.get("pl_result"), "plackett_luce")
        else:
            K = st.slider("Latent ranking classes", 2, 5, 2, key="plmix_k")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="plmix_seed")
            if st.button("RUN PLACKETT-LUCE MIXTURE", type="primary", key="run_plmix"):
                _run_and_store("plmix_result", plackett_luce_mixture, df=df, rank_columns=cols, components=int(K), seed=int(seed))
            _render_result(st.session_state.get("plmix_result"), "plackett_luce_mixture")

    with tabs[6]:
        method = st.selectbox("Categorical latent method", ["MCA + Ward", "Latent class analysis"], key="conf_lat_method")
        cats = st.multiselect("Categorical variables", all_cols, key="lat_cols")
        if method == "MCA + Ward":
            dims = st.slider("MCA dimensions retained", 2, 20, 5, key="mca_dims")
            clusters = st.slider("Ward clusters", 2, 10, 3, key="mca_clusters")
            if st.button("RUN MCA + WARD", type="primary", key="run_mca"):
                _run_and_store("mca_result", mca_ward, df=df, categorical_columns=cats, dimensions=int(dims), clusters=int(clusters))
            _render_result(st.session_state.get("mca_result"), "mca_ward")
        else:
            K = st.slider("Latent classes", 2, 8, 3, key="lca_k")
            starts = st.slider("Random starts", 1, 20, 5, key="lca_starts")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="lca_seed")
            if st.button("RUN LATENT CLASS ANALYSIS", type="primary", key="run_lca"):
                _run_and_store("lca_result", latent_class_analysis, df=df, categorical_columns=cats, classes=int(K), seed=int(seed), n_init=int(starts))
            _render_result(st.session_state.get("lca_result"), "latent_class_analysis")

    with tabs[7]:
        method = st.selectbox("Rare / ultra-rare utility", ["Dunn post-hoc", "TOST equivalence", "Meta-analysis", "Meta-regression (random effects)", "Rasch 1PL", "Local linear regression discontinuity", "Conditional logistic", "Exact 2x2: Fisher/Barnard/Boschloo", "Mantel-Haenszel", "McNemar exact", "Bowker symmetry", "Cochran Q", "Page L trend", "Quade rank test", "Jonckheere-Terpstra", "Brunner-Munzel", "Alexander-Govern", "Partial correlation", "Distance correlation", "Energy two-sample", "Heckman two-step", "Synthetic control", "Raw-normalised parsing audit"], key="conf_rare_method")
        if method == "Dunn post-hoc":
            value = st.selectbox("Numeric outcome", numeric, key="dunn_value")
            group = st.selectbox("Group", [c for c in all_cols if c != value], key="dunn_group")
            adj = st.selectbox("Adjustment", ["holm", "fdr_bh", "bonferroni"], key="dunn_adj")
            if st.button("RUN DUNN POST-HOC", type="primary", key="run_dunn"):
                _run_and_store("dunn_result", dunn_posthoc, df=df, value=value, group=group, adjustment=adj)
            _render_result(st.session_state.get("dunn_result"), "dunn_posthoc")
        elif method == "TOST equivalence":
            paired = st.checkbox("Paired samples", value=True, key="tost_paired")
            a = st.selectbox("Variable A / outcome", numeric, key="tost_a")
            if paired:
                b = st.selectbox("Variable B", [c for c in numeric if c != a], key="tost_b")
                group = None
            else:
                b = None
                group = st.selectbox("Two-level grouping variable", all_cols, key="tost_group")
            c1,c2=st.columns(2)
            with c1: low=st.number_input("Lower equivalence bound",value=-0.2,key="tost_low")
            with c2: high=st.number_input("Upper equivalence bound",value=0.2,key="tost_high")
            if st.button("RUN TOST", type="primary", key="run_tost"):
                _run_and_store("tost_result", equivalence_tost, df=df, value_a=a, value_b=b, group=group, low=float(low), high=float(high), paired=paired)
            _render_result(st.session_state.get("tost_result"), "equivalence_tost")
        elif method == "Meta-analysis":
            effect = st.selectbox("Study effect estimate", numeric, key="meta_eff")
            se = st.selectbox("Standard error", [c for c in numeric if c != effect], key="meta_se")
            label = st.selectbox("Study label (optional)", [None]+all_cols, key="meta_label")
            if st.button("RUN META-ANALYSIS", type="primary", key="run_meta"):
                _run_and_store("meta_result", meta_analysis, df=df, effect=effect, standard_error=se, study_label=label)
            _render_result(st.session_state.get("meta_result"), "meta_analysis")
        elif method == "Rasch 1PL":
            items = st.multiselect("Binary 0/1 item columns", numeric, key="rasch_items")
            if st.button("RUN RASCH 1PL", type="primary", key="run_rasch"):
                _run_and_store("rasch_result", rasch_1pl, df=df, item_columns=items)
            _render_result(st.session_state.get("rasch_result"), "rasch_1pl")
        elif method == "Local linear regression discontinuity":
            y = st.selectbox("Outcome", numeric, key="rdd_y")
            running = st.selectbox("Running / assignment variable", [c for c in numeric if c != y], key="rdd_running")
            cutoff = st.number_input("Cutoff", value=0.0, key="rdd_cutoff")
            bandwidth = st.number_input("Bandwidth", min_value=1e-12, value=1.0, key="rdd_bandwidth")
            covs = st.multiselect("Numeric covariates (optional)", [c for c in numeric if c not in {y,running}], key="rdd_covs")
            kernel = st.selectbox("Kernel", ["triangular", "uniform"], key="rdd_kernel")
            if st.button("RUN LOCAL LINEAR RDD", type="primary", key="run_rdd"):
                _run_and_store("rdd_result", regression_discontinuity, df=df, y=y, running=running, cutoff=float(cutoff), bandwidth=float(bandwidth), covariates=covs, kernel=kernel)
            _render_result(st.session_state.get("rdd_result"), "regression_discontinuity")
        elif method == "Conditional logistic":
            y = st.selectbox("Binary outcome", numeric, key="clogit_y")
            strata = st.selectbox("Matched set / stratum", all_cols, key="clogit_strata")
            xs = st.multiselect("Predictors (numeric or pre-encoded)", [c for c in numeric if c != y], key="clogit_x")
            if st.button("RUN CONDITIONAL LOGISTIC", type="primary", key="run_clogit"):
                _run_and_store("clogit_result", conditional_logistic, df=df, y=y, x_vars=xs, strata=strata)
            _render_result(st.session_state.get("clogit_result"), "conditional_logistic")
        elif method.startswith("Exact 2x2"):
            outcome = st.selectbox("Binary outcome", numeric, key="ex2_y")
            exposure = st.selectbox("Binary exposure", [c for c in numeric if c != outcome], key="ex2_x")
            if st.button("RUN EXACT 2x2 SUITE", type="primary", key="run_ex2"):
                _run_and_store("ex2_result", exact_2x2_tests, df=df, outcome=outcome, exposure=exposure)
            _render_result(st.session_state.get("ex2_result"), "exact_2x2")
        elif method == "Mantel-Haenszel":
            outcome = st.selectbox("Binary outcome", numeric, key="mh_y")
            exposure = st.selectbox("Binary exposure", [c for c in numeric if c != outcome], key="mh_x")
            strata = st.selectbox("Stratification variable", all_cols, key="mh_s")
            if st.button("RUN MANTEL-HAENSZEL", type="primary", key="run_mh"):
                _run_and_store("mh_result", mantel_haenszel, df=df, outcome=outcome, exposure=exposure, strata=strata)
            _render_result(st.session_state.get("mh_result"), "mantel_haenszel")
        elif method == "Page L trend":
            cols = st.multiselect("Ordered repeated-measure columns (left to right = hypothesised trend)", numeric, key="page_cols")
            ranked = st.checkbox("Inputs are already within-block ranks", value=False, key="page_ranked")
            if st.button("RUN PAGE L TREND TEST", type="primary", key="run_page"):
                _run_and_store("page_result", page_trend, df=df, columns=cols, ranked=ranked)
            _render_result(st.session_state.get("page_result"), "page_trend")
        elif method == "Alexander-Govern":
            value = st.selectbox("Numeric outcome", numeric, key="ag_value")
            group = st.selectbox("Group", all_cols, key="ag_group")
            if st.button("RUN ALEXANDER-GOVERN", type="primary", key="run_ag"):
                _run_and_store("ag_result", alexander_govern_test, df=df, value=value, group=group)
            _render_result(st.session_state.get("ag_result"), "alexander_govern")
        elif method == "Heckman two-step":
            y = st.selectbox("Outcome observed when selected", numeric, key="heck_y")
            selection = st.selectbox("Selection indicator 0/1", [c for c in numeric if c != y], key="heck_sel")
            outx = st.multiselect("Outcome-equation predictors", [c for c in all_cols if c not in {y,selection}], key="heck_outx")
            selx = st.multiselect("Selection-equation predictors", [c for c in all_cols if c not in {y,selection}], key="heck_selx")
            cats = st.multiselect("Categorical predictors appearing in either equation", list(dict.fromkeys(outx+selx)), key="heck_cat")
            if st.button("RUN HECKMAN TWO-STEP", type="primary", key="run_heck"):
                _run_and_store("heck_result", heckman_two_step, df=df, y=y, selection=selection, outcome_predictors=outx, selection_predictors=selx, categorical=cats)
            _render_result(st.session_state.get("heck_result"), "heckman_two_step")
        elif method == "Synthetic control":
            unit = st.selectbox("Panel unit", all_cols, key="sc_unit")
            time = st.selectbox("Time", [c for c in all_cols if c != unit], key="sc_time")
            outcome = st.selectbox("Outcome", numeric, key="sc_outcome")
            units = list(pd.Series(df[unit].dropna()).drop_duplicates())
            treated = st.selectbox("Treated unit", units, key="sc_treated") if units else None
            times = list(pd.Series(df[time].dropna()).drop_duplicates())
            intervention = st.selectbox("Intervention time", times, key="sc_intervention") if times else None
            if st.button("RUN SYNTHETIC CONTROL", type="primary", key="run_sc"):
                _run_and_store("sc_result", synthetic_control, df=df, unit=unit, time=time, outcome=outcome, treated_unit=treated, intervention_time=intervention)
            _render_result(st.session_state.get("sc_result"), "synthetic_control")
        elif method == "Meta-regression (random effects)":
            effect = st.selectbox("Study effect estimate", numeric, key="metareg_eff")
            se = st.selectbox("Standard error", [c for c in numeric if c != effect], key="metareg_se")
            preds = st.multiselect("Moderator variables", [c for c in all_cols if c not in {effect,se}], key="metareg_x")
            cats = st.multiselect("Categorical moderators", preds, key="metareg_cat")
            if st.button("RUN RANDOM-EFFECTS META-REGRESSION", type="primary", key="run_metareg"):
                _run_and_store("metareg_result", meta_regression, df=df, effect=effect, standard_error=se, predictors=preds, categorical=cats)
            _render_result(st.session_state.get("metareg_result"), "meta_regression")
        elif method == "McNemar exact":
            a = st.selectbox("Paired binary variable A", numeric, key="mcn_a")
            b = st.selectbox("Paired binary variable B", [c for c in numeric if c != a], key="mcn_b")
            exact = st.checkbox("Exact binomial test", value=True, key="mcn_exact")
            if st.button("RUN MCNEMAR", type="primary", key="run_mcn"):
                _run_and_store("mcn_result", mcnemar_test, df=df, variable_a=a, variable_b=b, exact=exact)
            _render_result(st.session_state.get("mcn_result"), "mcnemar")
        elif method == "Bowker symmetry":
            a = st.selectbox("Paired categorical variable A", all_cols, key="bow_a")
            b = st.selectbox("Paired categorical variable B", [c for c in all_cols if c != a], key="bow_b")
            if st.button("RUN BOWKER SYMMETRY", type="primary", key="run_bow"):
                _run_and_store("bow_result", bowker_symmetry_test, df=df, variable_a=a, variable_b=b)
            _render_result(st.session_state.get("bow_result"), "bowker_symmetry")
        elif method == "Cochran Q":
            cols = st.multiselect("Matched binary 0/1 conditions", numeric, key="cq_cols")
            if st.button("RUN COCHRAN Q", type="primary", key="run_cq"):
                _run_and_store("cq_result", cochran_q_test, df=df, columns=cols)
            _render_result(st.session_state.get("cq_result"), "cochran_q")
        elif method == "Quade rank test":
            cols = st.multiselect("Complete repeated conditions", numeric, key="quade_cols")
            if st.button("RUN QUADE TEST", type="primary", key="run_quade"):
                _run_and_store("quade_result", quade_test, df=df, columns=cols)
            _render_result(st.session_state.get("quade_result"), "quade_test")
        elif method == "Jonckheere-Terpstra":
            value = st.selectbox("Numeric outcome", numeric, key="jt_value")
            group = st.selectbox("Ordered group", all_cols, key="jt_group")
            alternative = st.selectbox("Ordered alternative", ["increasing", "decreasing", "two-sided"], key="jt_alt")
            perms = st.number_input("Permutations", min_value=99, max_value=99999, value=1999, step=100, key="jt_perm")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="jt_seed")
            st.caption("Group order follows sorted values unless you run the JSON recipe with an explicit order list.")
            if st.button("RUN JONCKHEERE-TERPSTRA", type="primary", key="run_jt"):
                _run_and_store("jt_result", jonckheere_terpstra, df=df, value=value, group=group, alternative=alternative, permutations=int(perms), seed=int(seed))
            _render_result(st.session_state.get("jt_result"), "jonckheere_terpstra")
        elif method == "Brunner-Munzel":
            value = st.selectbox("Numeric outcome", numeric, key="bm_value")
            group = st.selectbox("Two-level group", all_cols, key="bm_group")
            alt = st.selectbox("Alternative", ["two-sided", "greater", "less"], key="bm_alt")
            if st.button("RUN BRUNNER-MUNZEL", type="primary", key="run_bm"):
                _run_and_store("bm_result", brunner_munzel_test, df=df, value=value, group=group, alternative=alt)
            _render_result(st.session_state.get("bm_result"), "brunner_munzel")
        elif method == "Partial correlation":
            x = st.selectbox("X", numeric, key="pcor_x")
            y = st.selectbox("Y", [c for c in numeric if c != x], key="pcor_y")
            ctrls = st.multiselect("Numeric controls", [c for c in numeric if c not in {x,y}], key="pcor_ctrl")
            pm = st.selectbox("Correlation type", ["pearson", "spearman"], key="pcor_method")
            if st.button("RUN PARTIAL CORRELATION", type="primary", key="run_pcor"):
                _run_and_store("pcor_result", partial_correlation, df=df, x=x, y=y, controls=ctrls, method=pm)
            _render_result(st.session_state.get("pcor_result"), "partial_correlation")
        elif method == "Distance correlation":
            xs = st.multiselect("X variable set", numeric, key="dcor_x")
            ys = st.multiselect("Y variable set", [c for c in numeric if c not in xs], key="dcor_y")
            perms = st.number_input("Permutations", min_value=99, max_value=99999, value=999, step=100, key="dcor_perm")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="dcor_seed")
            if st.button("RUN DISTANCE CORRELATION", type="primary", key="run_dcor"):
                _run_and_store("dcor_result", distance_correlation_test, df=df, x_columns=xs, y_columns=ys, permutations=int(perms), seed=int(seed))
            _render_result(st.session_state.get("dcor_result"), "distance_correlation")
        elif method == "Energy two-sample":
            cols = st.multiselect("Numeric multivariate outcome", numeric, key="energy_cols")
            group = st.selectbox("Two-level group", all_cols, key="energy_group")
            perms = st.number_input("Permutations", min_value=99, max_value=99999, value=999, step=100, key="energy_perm")
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, key="energy_seed")
            if st.button("RUN ENERGY TWO-SAMPLE TEST", type="primary", key="run_energy"):
                _run_and_store("energy_result", energy_two_sample_test, df=df, columns=cols, group=group, permutations=int(perms), seed=int(seed))
            _render_result(st.session_state.get("energy_result"), "energy_two_sample")
        else:
            raw = st.selectbox("Raw / free-text source column", all_cols, key="parse_raw")
            norm = st.selectbox("Normalised numeric column", [c for c in all_cols if c != raw], key="parse_norm")
            flag = st.selectbox("Parsing-flag column (optional)", [None]+[c for c in all_cols if c not in {raw,norm}], key="parse_flag")
            direct_text = st.text_input("Flags counted as direct values (comma-separated)", value="numeric_factor", key="parse_direct")
            direct = [x.strip() for x in direct_text.split(",") if x.strip()]
            if st.button("RUN RAW-NORMALISED SAMPLE AUDIT", type="primary", key="run_parse_audit"):
                _run_and_store("parse_result", parsed_numeric_audit, df=df, raw_column=raw, normalised_column=norm, flag_column=flag, direct_flags=direct)
            _render_result(st.session_state.get("parse_result"), "parsed_numeric_audit")


    with tabs[8]:
        st.markdown("### Portable analysis recipes")
        st.caption("This is the future-proofing layer: save a method + variable mapping as plain JSON, then rerun it on any compatible future dataset without adding paper-specific code.")
        method_key = st.selectbox("Recipe method", sorted(RUNNERS), key="recipe_method_key")
        template = recipe_json(method_key)
        st.download_button("Download JSON recipe template", template.encode("utf-8"), f"{method_key}_recipe.json", "application/json", key="recipe_template_download")
        payload = st.text_area("Recipe JSON (edit variable names/options, then run)", value=template, height=420, key=f"recipe_editor_{method_key}")
        if st.button("RUN RECIPE", type="primary", key="run_recipe_json"):
            try:
                with st.spinner("Running portable recipe..."):
                    result = run_recipe(df, payload)
                    st.session_state["portable_recipe_result"] = result
                st.success("Recipe executed successfully.")
            except Exception as exc:
                st.error(f"Recipe failed: {exc}")
        _render_result(st.session_state.get("portable_recipe_result"), "portable_analysis_recipe")
