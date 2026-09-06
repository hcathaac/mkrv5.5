"""Streamlit interface for standalone Agentic Research Mode v5.8.1."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agentic_research import (
    agent_context_text,
    agentic_submission_package,
    build_agentic_plan,
    extract_source_evidence,
    generate_questions_with_ai,
    generate_research_questions,
    RQ_RESPONSE_SCHEMA,
    SYNTHESIS_RESPONSE_SCHEMA,
    literature_key_terms,
    offline_agent_reply,
    ollama_agent_reply,
    ollama_text_reply,
    refine_run_with_ai,
    run_agentic_workflow,
)
from llm_bridge import configured as llm_configured, llm_reply
from research_chair import extract_pdf_collection, ollama_models


def _external_agent_reply(run, question: str, config: dict, history: list[dict[str, str]]) -> str:
    evidence = agent_context_text(run, question, max_chars=55000)
    recent = "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-8:])
    prompt = f"""RECENT CONVERSATION
{recent}

COMPUTED AND SOURCE EVIDENCE
{evidence}

USER QUESTION
{question}
"""
    system = (
        "You are the Makryvelios Agentic Research Mode. Answer the user's exact question rather than returning a generic research-safety paragraph. "
        "Use only the supplied computed results and uploaded-PDF evidence. Preserve all numerical values exactly. If the evidence cannot answer the question, identify precisely what is missing. "
        "Distinguish descriptive association, adjusted association, prediction, optimisation and causal estimates. Cite uploaded literature by document and page where available. "
        "Answer in the same language as the user unless asked otherwise."
    )
    return llm_reply(prompt, config, system=system)


def _selected_agent_engine() -> tuple[str, str, str]:
    options = [
        "Smart offline semantic agent — no AI/API",
        "Local AI — Ollama (no API key)",
        "Configured AI — sidebar provider",
    ]
    default_index = 2 if llm_configured(st.session_state.get("llm_config", {})) else 0
    engine = st.radio(
        "Agent intelligence engine",
        options,
        index=default_index,
        horizontal=True,
        key="agentic_intelligence_engine",
        help="Numerical models always run in the deterministic application engines. This setting controls language understanding, research-question generation, full-draft synthesis and conversation.",
    )
    endpoint = "http://127.0.0.1:11434"
    model = ""
    if engine.startswith("Local AI"):
        a, b = st.columns([2, 2])
        with a:
            endpoint = st.text_input("Ollama endpoint", value="http://127.0.0.1:11434", key="agentic_ollama_endpoint")
        with b:
            detected = ollama_models(endpoint=endpoint, timeout=0.6)
            if detected:
                model = st.selectbox("Local model", detected, key="agentic_ollama_model")
                st.success(f"Local AI detected: {model}")
            else:
                model = st.text_input("Local model name", value="llama3.1:8b", key="agentic_ollama_model_manual")
                st.info("No Ollama model was auto-detected at this endpoint. You can still enter a model name and retry after Ollama is running.")
        st.caption("For a locally deployed Streamlit app, Ollama can run on the same computer with no API key. Streamlit Community Cloud cannot reach your laptop's localhost; use a reachable/self-hosted Ollama endpoint there.")
    elif engine.startswith("Configured AI"):
        config = st.session_state.get("llm_config", {})
        if llm_configured(config):
            st.success(f"External model ready: {config.get('provider')} · {config.get('model')}")
        else:
            st.warning("Configure Gemini, Groq, Claude, Ollama or another compatible model in the persistent sidebar AI panel. Until then, switch to Smart offline or direct Local AI.")
    return engine, endpoint, model


def render_agentic_research(df: pd.DataFrame, selected_label: str = "") -> None:
    st.subheader("Agentic Research Mode — evidence-aware research intelligence")
    st.markdown(
        '<div class="guide"><b>Different from Research Chair.</b> This module maintains a research conversation, semantically retrieves the exact computed rows/PDF passages relevant to the question, and can use a local language model with no API key.<br>'
        '<b>Three intelligence levels.</b> Smart deterministic semantic routing, direct Local Ollama, or the configured sidebar AI engine (Gemini/Groq/Claude/Ollama/OpenAI-compatible). Numerical analysis remains deterministic in all three modes.<br>'
        '<b>End-to-end.</b> Dataset + literature → specific RQs → approval-gated model execution → evidence-grounded conversation → DOCX/XLSX/JSON/HTML/graphics submission package.</div>',
        unsafe_allow_html=True,
    )
    st.warning("The agent may automate research labour, but it does not invent evidence. Bibliographic details, quotations, causal identification and final institutional/journal compliance still require verification.")

    engine, ollama_endpoint, ollama_model = _selected_agent_engine()
    ai_engine_selected = engine.startswith("Local AI") or engine.startswith("Configured AI")
    auto_ai_synthesis = st.checkbox(
        "Automatically run an AI synthesis pass after the deterministic analysis",
        value=ai_engine_selected,
        disabled=not ai_engine_selected,
        key="agentic_auto_ai_synthesis",
        help="This rewrites Abstract, Results, Discussion, Conclusion and run-specific Limitations from the computed tables and retrieved PDF evidence. It never changes the numerical results.",
    )
    if ai_engine_selected:
        st.caption("AI synthesis is a second pass over computed evidence, not a substitute for the statistical/optimisation engines. The submission DOCX/HTML will use the refined sections when available.")

    pdf_uploads = st.file_uploader(
        "Literature PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="agentic_pdfs",
        help="Text is extracted locally. Source evidence retains document and page provenance.",
    )
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
            st.dataframe(evidence.head(500), width="stretch", hide_index=True)
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
    st.markdown("### Variable roles")
    if df.empty:
        st.info("No spreadsheet dataset is active. Literature intelligence and conversation can still work; numerical execution requires a dataset.")
        outcome = None
        predictors = []
        group = None
        time_col = None
        region = None
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            outcome = st.selectbox("Primary outcome (optional)", [None] + numeric, key="agentic_outcome")
        with c2:
            predictor_options = [c for c in numeric if c != outcome]
            predictors = st.multiselect("Primary predictors", predictor_options, default=predictor_options[:min(6, len(predictor_options))], max_selections=60, key="agentic_predictors")
        with c3:
            group_candidates = [c for c in all_columns if c != outcome and 2 <= df[c].nunique(dropna=True) <= 80]
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

    st.markdown("### Specific research-question generator")
    rq_count = st.slider("Questions to generate in one batch", 10, 200, 150, 10, key="agentic_rq_count")
    st.caption("Smart offline mode ranks actual observed relationships before composing questions. Local/API AI modes additionally read the schema, observed correlation leads and uploaded PDF evidence to create less templated questions.")
    if st.button("GENERATE SPECIFIC RESEARCH QUESTIONS", key="agentic_generate_rqs"):
        try:
            with st.spinner(f"Generating up to {int(rq_count)} grounded research questions..."):
                if engine.startswith("Local AI"):
                    if not ollama_model.strip():
                        raise ValueError("Enter or select a local Ollama model first.")
                    reply_fn = lambda prompt: ollama_text_reply(prompt, ollama_model, endpoint=ollama_endpoint, timeout=240, temperature=0.18)
                    rqs = generate_questions_with_ai(df, pdf_pages, goal, int(rq_count), reply_fn, batch_size=25)
                    source = f"Local Ollama · {ollama_model}"
                elif engine.startswith("Configured AI"):
                    config = st.session_state.get("llm_config", {})
                    if not llm_configured(config):
                        raise ValueError("Configure the external LLM in the sidebar or select another engine.")
                    system = "Generate only grounded, specific research questions from the supplied schema, observed relationship leads and PDF evidence. Never invent variables or source claims. Return the requested structured output only."
                    provider_name = str(config.get("provider", ""))
                    schema = RQ_RESPONSE_SCHEMA if provider_name.lower().startswith(("google", "gemini")) else None
                    reply_fn = lambda prompt: llm_reply(prompt, config, system=system, timeout=150, response_schema=schema)
                    rqs = generate_questions_with_ai(df, pdf_pages, goal, int(rq_count), reply_fn, batch_size=20)
                    source = f"Configured AI · {config.get('provider')} · {config.get('model')}"
                else:
                    rqs = generate_research_questions(df, pdf_pages, limit=int(rq_count))
                    source = "Smart deterministic data-aware generator"
            st.session_state["agentic_rqs"] = rqs
            ai_count = int(getattr(rqs, "attrs", {}).get("ai_generated", len(rqs) if ai_engine_selected else 0))
            recovery_count = int(getattr(rqs, "attrs", {}).get("deterministic_recovery", 0))
            parse_failures = int(getattr(rqs, "attrs", {}).get("parse_failures", 0))
            source_detail = source
            if recovery_count:
                source_detail += f" + deterministic recovery ({recovery_count})"
            st.session_state["agentic_rq_source"] = source_detail
            st.success(f"Generated {len(rqs)} research questions. AI-grounded: {ai_count}; deterministic recovery: {recovery_count}.")
            if parse_failures:
                st.info(f"The selected model returned {parse_failures} malformed structured-response batch(es); the agent repaired/retried them automatically instead of failing the workflow.")
        except Exception as exc:
            st.error(f"Question generation failed: {exc}")
    rqs = st.session_state.get("agentic_rqs")
    if isinstance(rqs, pd.DataFrame):
        st.caption(f"Question engine: {st.session_state.get('agentic_rq_source', 'current workflow')}")
        st.dataframe(rqs, width="stretch", hide_index=True, height=540)
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
        st.dataframe(pd.DataFrame(plan.steps), width="stretch", hide_index=True)
        for warning in plan.warnings:
            st.info(warning)
        approval = st.checkbox("I have reviewed the mapped variables and approve this bounded workflow", key="agentic_approval")
        if st.button("APPROVE & RUN AGENTIC WORKFLOW", type="primary", disabled=not approval or (df.empty and pdf_pages.empty), key="agentic_run"):
            try:
                with st.spinner("Running the approved deterministic research workflow..."):
                    run = run_agentic_workflow(df, plan=plan, pdf_pages=pdf_pages, outcome=outcome, predictors=predictors, group=group, question_limit=int(rq_count))
                chosen_rqs = st.session_state.get("agentic_rqs")
                if isinstance(chosen_rqs, pd.DataFrame) and not chosen_rqs.empty:
                    run.tables["Research questions"] = chosen_rqs.copy()
                    run.manifest["research_questions_generated"] = int(len(chosen_rqs))
                    run.manifest["research_question_engine"] = st.session_state.get("agentic_rq_source", "pre-generated")
                if auto_ai_synthesis and ai_engine_selected:
                    try:
                        with st.spinner("Running evidence-grounded AI synthesis over the completed run..."):
                            if engine.startswith("Local AI"):
                                if not ollama_model.strip():
                                    raise ValueError("Enter or select a local Ollama model first.")
                                reply_fn = lambda prompt: ollama_text_reply(prompt, ollama_model, endpoint=ollama_endpoint, timeout=240, temperature=0.10)
                                provider_label = f"Local Ollama · {ollama_model}"
                            else:
                                config = st.session_state.get("llm_config", {})
                                if not llm_configured(config):
                                    raise ValueError("Configure a sidebar AI provider first.")
                                system = "You are an evidence-grounded research synthesis engine. Preserve every computed number exactly, name the actual variables/models/sources, and never replace missing evidence with generic boilerplate."
                                reply_fn = lambda prompt: llm_reply(prompt, config, system=system, timeout=240, response_schema=SYNTHESIS_RESPONSE_SCHEMA)
                                provider_label = f"{config.get('provider')} · {config.get('model')}"
                            run = refine_run_with_ai(run, reply_fn, provider_label=provider_label)
                    except Exception as synth_exc:
                        st.warning(f"Deterministic analysis completed, but the optional AI synthesis pass failed: {synth_exc}. The offline results were retained unchanged.")
                st.session_state["agentic_run_result"] = run
                st.session_state["agentic_chat_history"] = []
                st.success("Agentic workflow completed. The run is now available to the evidence-aware research conversation and submission builder.")
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
        ai_meta = run.manifest.get("ai_synthesis", {})
        if ai_meta.get("enabled"):
            st.success(f"AI synthesis active: {ai_meta.get('provider')}. Numerical tables remain deterministic and unchanged.")
        else:
            st.info("This run currently uses the deterministic first-pass narrative. Choose Local AI or a configured sidebar AI engine and run the synthesis pass below for a non-template research draft.")
        if run.narratives.get("abstract"):
            st.markdown("#### Abstract")
            st.write(run.narratives.get("abstract", ""))
        st.markdown("#### Results synthesis")
        st.write(run.narratives.get("results", run.narratives.get("discussion", "")))
        st.markdown("#### Discussion")
        st.write(run.narratives.get("discussion", ""))
        st.markdown("#### Conclusion")
        st.write(run.narratives.get("conclusion", ""))

        can_refine = ai_engine_selected
        if st.button("REFINE / REWRITE ENTIRE DRAFT WITH SELECTED AI", type="primary", disabled=not can_refine, key="agentic_refine_full_run"):
            try:
                with st.spinner("Retrieving computed rows and PDF evidence, then rewriting the draft..."):
                    if engine.startswith("Local AI"):
                        if not ollama_model.strip():
                            raise ValueError("Enter or select a local Ollama model first.")
                        reply_fn = lambda prompt: ollama_text_reply(prompt, ollama_model, endpoint=ollama_endpoint, timeout=240, temperature=0.10)
                        provider_label = f"Local Ollama · {ollama_model}"
                    else:
                        config = st.session_state.get("llm_config", {})
                        if not llm_configured(config):
                            raise ValueError("Configure a sidebar AI provider first.")
                        system = "You are an evidence-grounded research synthesis engine. Preserve every computed number exactly, name the actual variables/models/sources, and never replace missing evidence with generic boilerplate."
                        reply_fn = lambda prompt: llm_reply(prompt, config, system=system, timeout=240, response_schema=SYNTHESIS_RESPONSE_SCHEMA)
                        provider_label = f"{config.get('provider')} · {config.get('model')}"
                    run = refine_run_with_ai(run, reply_fn, provider_label=provider_label)
                    st.session_state["agentic_run_result"] = run
                st.success("AI synthesis completed. Abstract, Results, Discussion, Conclusion, limitations and the submission package now use the evidence-grounded rewrite.")
                st.rerun()
            except Exception as exc:
                st.error(f"AI synthesis failed: {exc}")

    with out_tabs[1]:
        for name, table in run.tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty and name not in {"Research questions", "Literature source evidence", "Literature key terms"}:
                with st.expander(name, expanded=name in {"Quality audit", "Top correlations", "OLS coefficients"}):
                    st.dataframe(table.head(2000), width="stretch", hide_index=True)
        for name, payload in run.interactive_html.items():
            st.download_button(f"Download {name}", payload, name, "text/html", key=f"agentic_interactive_{name}")

    with out_tabs[2]:
        st.dataframe(run.tables.get("Research questions", pd.DataFrame()), width="stretch", hide_index=True, height=650)

    with out_tabs[3]:
        st.dataframe(run.tables.get("Literature source evidence", pd.DataFrame()), width="stretch", hide_index=True, height=600)
        st.caption("Document/page provenance is retained. The agent may retrieve relevant passages, but quotations and final bibliographic metadata must be checked against the originals.")

    with out_tabs[4]:
        if run.narratives.get("abstract"):
            st.markdown("#### Abstract")
            st.write(run.narratives.get("abstract", ""))
        st.markdown("#### Results")
        st.write(run.narratives.get("results", run.narratives.get("discussion", "")))
        st.markdown("#### Discussion")
        st.write(run.narratives.get("discussion", ""))
        st.markdown("#### Conclusions")
        st.write(run.narratives.get("conclusion", ""))
        st.markdown("#### Limitations")
        st.write(run.narratives.get("limitations", ""))
        if run.narratives.get("offline_discussion"):
            with st.expander("Audit: original deterministic first-pass narrative", expanded=False):
                st.write(run.narratives.get("offline_discussion", ""))

    with out_tabs[5]:
        st.markdown("#### Evidence-aware research conversation")
        st.caption("This is now a single multi-turn agent conversation. Smart Offline semantically routes the question to actual result rows; Local/Configured AI receives only retrieved computed/source evidence and the recent conversation.")
        history = st.session_state.setdefault("agentic_chat_history", [])
        for message in history:
            with st.chat_message(message.get("role", "assistant")):
                st.markdown(message.get("content", ""))

        q1, q2, q3, q4 = st.columns(4)
        quick_question = None
        with q1:
            if st.button("Strongest finding", key="agentic_quick_strongest"):
                quick_question = "What is the strongest defensible finding in this run, with the exact evidence?"
        with q2:
            if st.button("Weakest finding", key="agentic_quick_weakest"):
                quick_question = "What is the weakest finding and what can I not safely conclude?"
        with q3:
            if st.button("Literature vs results", key="agentic_quick_lit"):
                quick_question = "Which uploaded literature evidence is most relevant to the strongest computed results, and where does it agree or not directly support them?"
        with q4:
            if st.button("Next analysis", key="agentic_quick_next"):
                quick_question = "Given the actual results and diagnostics, what should I run next and why?"

        typed = st.chat_input("Ask anything about this run, a variable, model, source, finding, limitation or next analysis")
        question = typed or quick_question
        if question:
            history.append({"role": "user", "content": question})
            try:
                with st.spinner("Reading the relevant computed evidence..."):
                    if engine.startswith("Local AI"):
                        if not ollama_model.strip():
                            raise ValueError("Select or enter a local Ollama model first.")
                        answer = ollama_agent_reply(run, question, ollama_model, endpoint=ollama_endpoint, history=history[:-1])
                        label = f"Local AI · {ollama_model}"
                    elif engine.startswith("Configured AI"):
                        config = st.session_state.get("llm_config", {})
                        if not llm_configured(config):
                            raise ValueError("Configure the external LLM in the sidebar or switch the Agent intelligence engine.")
                        answer = _external_agent_reply(run, question, config, history[:-1])
                        label = f"Configured AI · {config.get('provider')} · {config.get('model')}"
                    else:
                        answer = offline_agent_reply(run, question, history=history[:-1])
                        label = "Smart semantic offline agent"
                history.append({"role": "assistant", "content": answer, "engine": label})
                st.rerun()
            except Exception as exc:
                history.append({"role": "assistant", "content": f"Agent error: {exc}"})
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Current conversation engine: {engine}")
        with c2:
            if st.button("CLEAR RESEARCH CONVERSATION", key="agentic_clear_chat"):
                st.session_state["agentic_chat_history"] = []
                st.rerun()

    with out_tabs[6]:
        title = st.text_input("Draft paper title", value="Makryvelios Agentic Research Draft", key="agentic_paper_title")
        package = agentic_submission_package(run, title=title)
        st.success("Near-submission package assembled locally. It includes Word, Excel, JSON, HTML, 600-dpi/vector graphics and reproducibility material where generated.")
        st.download_button("DOWNLOAD COMPLETE AGENTIC SUBMISSION PACKAGE", package, "makryvelios_agentic_submission_package.zip", "application/zip", type="primary", key="agentic_package_download")
        st.json(run.manifest)
