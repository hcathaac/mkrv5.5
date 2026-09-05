"""Streamlit interface for standalone Agentic Research Mode v5.8.0."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agentic_research import (
    agentic_submission_package,
    build_agentic_plan,
    extract_source_evidence,
    generate_research_questions,
    literature_key_terms,
    offline_agent_reply,
    run_agentic_workflow,
)
from llm_bridge import configured as llm_configured, llm_reply
from research_chair import extract_pdf_collection


def _compact_evidence(run) -> str:
    chunks = [f"Goal: {run.plan.goal}"]
    for name in ["Quality audit", "Top correlations", "OLS coefficients", "OLS diagnostics", "Group tests", "PCA variance", "Cluster profiles", "Literature key terms", "Literature source evidence"]:
        table = run.tables.get(name)
        if isinstance(table, pd.DataFrame) and not table.empty:
            chunks.append(f"\n[{name}]\n" + table.head(20).to_csv(index=False))
    chunks.append("\nOffline discussion:\n" + run.narratives.get("discussion", ""))
    chunks.append("\nOffline conclusion:\n" + run.narratives.get("conclusion", ""))
    return "\n".join(chunks)[:50000]


def render_agentic_research(df: pd.DataFrame, selected_label: str = "") -> None:
    st.subheader("Agentic Research Mode — fast, reproducible end-to-end research runner")
    st.markdown(
        '<div class="guide"><b>Purpose.</b> Turn a research objective, active dataset and literature PDFs into a bounded research workflow, up to 150 candidate research questions, local evidence extraction, executed analysis, discussion/conclusions and a near-submission package.<br>'
        '<b>Offline first.</b> Planning, PDF text extraction, numerical analysis, question generation, drafting and exports work without an AI API.<br>'
        '<b>Optional AI.</b> If the sidebar LLM key is configured, the model can refine plans and prose after deterministic computation. It never becomes the numerical solver and never runs an unapproved workflow.</div>',
        unsafe_allow_html=True,
    )
    st.warning("Target: remove repetitive research labour and produce a near-submission analytical package. Final scholarly verification remains mandatory: references, quotations, causal claims, formatting and institutional/journal requirements must be checked before submission.")

    mode = st.radio("Agentic execution mode", ["Offline deterministic (no API required)", "Hybrid: deterministic + optional LLM synthesis"], horizontal=True, key="agentic_mode")
    pdf_uploads = st.file_uploader("Literature PDFs", type=["pdf"], accept_multiple_files=True, key="agentic_pdfs", help="PDF text is extracted locally. Image-only scans require OCR outside this module before text can be analysed.")
    if pdf_uploads:
        try:
            pdf_pages = extract_pdf_collection(tuple((item.name, item.getvalue()) for item in pdf_uploads))
            st.success(f"Indexed {len(pdf_uploads)} PDF(s), {len(pdf_pages):,} pages, {int(pdf_pages.characters.sum()):,} extracted characters.")
        except Exception as exc:
            st.error(f"PDF extraction failed: {exc}")
            pdf_pages = pd.DataFrame(columns=["document", "page", "text", "characters"])
    else:
        pdf_pages = pd.DataFrame(columns=["document", "page", "text", "characters"])

    if not pdf_pages.empty:
        with st.expander("Literature evidence index", expanded=False):
            evidence = extract_source_evidence(pdf_pages)
            terms = literature_key_terms(pdf_pages, 30)
            st.dataframe(evidence.head(300), width="stretch", hide_index=True)
            st.markdown("##### Frequent literature terms")
            st.dataframe(terms, width="stretch", hide_index=True)

    goal = st.text_area(
        "Principal research objective / instruction",
        value="Identify the strongest defensible empirical findings, test the most relevant relationships, assess robustness and prepare a publication-ready analytical draft while distinguishing association, prediction, optimisation and causality.",
        height=140,
        key="agentic_goal",
    )

    all_columns = list(df.columns)
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = [c for c in all_columns if c not in numeric]
    st.markdown("### Variable roles")
    if df.empty:
        st.info("No spreadsheet dataset is active. Literature indexing and LLM discussion can still be used, but numerical execution requires a dataset.")
        outcome = None; predictors = []; group = None; time_col = None; region = None
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            outcome = st.selectbox("Primary outcome (optional)", [None] + numeric, key="agentic_outcome")
        with c2:
            predictor_options = [c for c in numeric if c != outcome]
            predictors = st.multiselect("Primary predictors", predictor_options, default=predictor_options[:min(6, len(predictor_options))], max_selections=60, key="agentic_predictors")
        with c3:
            group_candidates = [c for c in all_columns if c != outcome and df[c].nunique(dropna=True) <= 80]
            group = st.selectbox("Group / subgroup (optional)", [None] + group_candidates, key="agentic_group")
        c4, c5 = st.columns(2)
        with c4:
            time_guess = next((c for c in all_columns if any(t in str(c).lower() for t in ("year", "date", "time"))), None)
            time_opts = [None] + all_columns
            time_col = st.selectbox("Time variable (optional)", time_opts, index=time_opts.index(time_guess) if time_guess in time_opts else 0, key="agentic_time")
        with c5:
            region_guess = next((c for c in all_columns if any(t in str(c).lower() for t in ("region", "nuts", "municip"))), None)
            region_opts = [None] + all_columns
            region = st.selectbox("Region / geography (optional)", region_opts, index=region_opts.index(region_guess) if region_guess in region_opts else 0, key="agentic_region")

    st.markdown("### Research-question generator")
    rq_count = st.slider("Questions to generate in one batch", 10, 200, 150, 10, key="agentic_rq_count")
    if st.button("GENERATE RESEARCH QUESTIONS", key="agentic_generate_rqs"):
        try:
            rqs = generate_research_questions(df, pdf_pages, limit=int(rq_count))
            st.session_state["agentic_rqs"] = rqs
            st.success(f"Generated {len(rqs)} candidate research questions from the active variables and uploaded literature terms.")
        except Exception as exc:
            st.error(f"Question generation failed: {exc}")
    rqs = st.session_state.get("agentic_rqs")
    if isinstance(rqs, pd.DataFrame):
        st.dataframe(rqs, width="stretch", hide_index=True, height=520)
        st.download_button("Download research-question bank (CSV)", rqs.to_csv(index=False).encode("utf-8-sig"), "agentic_research_questions.csv", "text/csv", key="agentic_rq_csv")

    st.markdown("### Research plan + approval gate")
    if st.button("BUILD AGENTIC PLAN", type="primary", key="agentic_build_plan"):
        try:
            plan = build_agentic_plan(df, goal=goal, outcome=outcome, predictors=predictors, group=group, time_column=time_col, region=region, pdf_pages=pdf_pages)
            st.session_state["agentic_plan"] = plan
        except Exception as exc:
            st.error(f"Plan generation failed: {exc}")
    plan = st.session_state.get("agentic_plan")
    if plan is not None:
        plan_table = pd.DataFrame(plan.steps)
        st.dataframe(plan_table, width="stretch", hide_index=True)
        for warning in plan.warnings:
            st.info(warning)
        approval = st.checkbox("I have reviewed the mapped variables and approve this bounded workflow", key="agentic_approval")
        if st.button("APPROVE & RUN AGENTIC WORKFLOW", type="primary", disabled=not approval or (df.empty and pdf_pages.empty), key="agentic_run"):
            try:
                with st.spinner("Running the approved offline research workflow..."):
                    run = run_agentic_workflow(df, plan=plan, pdf_pages=pdf_pages, outcome=outcome, predictors=predictors, group=group, question_limit=int(rq_count))
                st.session_state["agentic_run_result"] = run
                st.session_state["agentic_rqs"] = run.tables.get("Research questions")
                st.success("Agentic workflow completed. Numerical analysis, discussion, conclusions and package components are ready.")
            except Exception as exc:
                st.error(f"Agentic workflow failed: {exc}")

    run = st.session_state.get("agentic_run_result")
    if run is None:
        return

    out_tabs = st.tabs(["Executive synthesis", "All analytical outputs", "150 RQs", "Literature evidence", "Discussion & conclusions", "Research conversation", "Submission package"])
    with out_tabs[0]:
        a, b, c, d = st.columns(4)
        a.metric("Rows analysed", f"{run.manifest.get('rows', 0):,}")
        b.metric("Tables generated", len(run.tables))
        c.metric("Research questions", run.manifest.get("research_questions_generated", 0))
        d.metric("PDF sources", len(run.manifest.get("pdf_documents", [])))
        st.markdown("#### Offline evidence-grounded synthesis")
        st.write(run.narratives.get("discussion", ""))
        st.markdown("#### Conclusion")
        st.write(run.narratives.get("conclusion", ""))

    with out_tabs[1]:
        for name, table in run.tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty and name not in {"Research questions", "Literature source evidence", "Literature key terms"}:
                with st.expander(name, expanded=name in {"Quality audit", "Top correlations", "OLS coefficients"}):
                    st.dataframe(table.head(2000), width="stretch", hide_index=True)
        for name, payload in run.interactive_html.items():
            st.download_button(f"Download {name}", payload, name, "text/html", key=f"agentic_interactive_{name}")

    with out_tabs[2]:
        rqs = run.tables.get("Research questions", pd.DataFrame())
        st.dataframe(rqs, width="stretch", hide_index=True, height=650)

    with out_tabs[3]:
        st.dataframe(run.tables.get("Literature source evidence", pd.DataFrame()), width="stretch", hide_index=True, height=600)
        st.caption("The source index preserves document/page provenance and detected DOI/URL strings. It does not invent missing author/title metadata and must be verified before citation.")

    with out_tabs[4]:
        st.markdown("#### Discussion")
        st.write(run.narratives.get("discussion", ""))
        st.markdown("#### Conclusions")
        st.write(run.narratives.get("conclusion", ""))
        st.markdown("#### Limitations")
        st.write(run.narratives.get("limitations", ""))

    with out_tabs[5]:
        st.markdown("#### Offline research conversation")
        offline_q = st.text_input("Ask the computed run", value="What is the strongest finding and what can I safely conclude?", key="agentic_offline_question")
        if st.button("ASK OFFLINE AGENT", key="agentic_offline_ask"):
            st.session_state["agentic_offline_answer"] = offline_agent_reply(run, offline_q)
        if st.session_state.get("agentic_offline_answer"):
            st.markdown("##### DETERMINISTIC ANSWER — no external AI used")
            st.write(st.session_state["agentic_offline_answer"])

        st.divider()
        st.markdown("#### Optional LLM synthesis")
        config = st.session_state.get("llm_config", {})
        if not mode.startswith("Hybrid"):
            st.info("The offline conversation above remains available. Switch to Hybrid only if you also want external LLM synthesis.")
        elif not llm_configured(config):
            st.info("Enter an LLM API key/model in the persistent sidebar panel. The offline agent and all numerical analysis remain fully usable without it.")
        else:
            st.success(f"LLM available for synthesis: {config.get('provider')} · {config.get('model')}")
            prompt = st.text_area("Discuss the computed run with the LLM", value="Review the computed evidence, identify the strongest defensible findings, challenge weak interpretations, propose the next analyses, and draft a concise Discussion and Conclusion suitable for a research paper. Do not change any numerical result and do not claim causality unless the evidence explicitly contains a causal estimate with stated assumptions.", height=180, key="agentic_llm_prompt")
            if st.button("ASK LLM ABOUT THE COMPUTED RUN", key="agentic_llm_ask"):
                try:
                    evidence = _compact_evidence(run)
                    answer = llm_reply(f"USER TASK:\n{prompt}\n\nCOMPUTED EVIDENCE:\n{evidence}", config)
                    st.session_state["agentic_llm_answer"] = answer
                except Exception as exc:
                    st.error(f"LLM synthesis failed: {exc}")
            if st.session_state.get("agentic_llm_answer"):
                st.markdown("##### AI SYNTHESIS — not a computed statistical output")
                st.markdown(st.session_state["agentic_llm_answer"])

    with out_tabs[6]:
        title = st.text_input("Draft paper title", value="Makryvelios Agentic Research Draft", key="agentic_paper_title")
        package = agentic_submission_package(run, title=title)
        st.success("Near-submission package assembled locally. It includes Word, Excel, JSON, HTML, 600-dpi/vector graphics and reproducibility material where generated.")
        st.download_button("DOWNLOAD COMPLETE AGENTIC SUBMISSION PACKAGE", package, "makryvelios_agentic_submission_package.zip", "application/zip", type="primary", key="agentic_package_download")
        st.json(run.manifest)
