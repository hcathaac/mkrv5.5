"""Agentic Research Mode backend for Makryvelios v5.8.0.

The agent has two layers:
1) deterministic/offline planning, evidence indexing, analysis and drafting;
2) optional LLM synthesis invoked explicitly by the UI.

No external model is required for numerical execution or package generation.
"""
from __future__ import annotations

import html
import io
import json
import math
import re
import zipfile
from collections import Counter
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analytics_core import (
    categorical_summary,
    cluster_table,
    correlation_matrix,
    descriptive_statistics,
    fit_detailed_model,
    group_tests,
    pca_table,
    quality_summary,
    to_excel_bytes,
)
from research_chair import docx_bytes


@dataclass
class AgenticPlan:
    goal: str
    steps: list[dict[str, str]]
    warnings: list[str]
    mappings: dict[str, Any]


@dataclass
class AgenticRun:
    plan: AgenticPlan
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    narratives: dict[str, str] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    interactive_html: dict[str, bytes] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def extract_source_evidence(pdf_pages: pd.DataFrame) -> pd.DataFrame:
    """Create a page-level literature/source index without inventing bibliography metadata."""
    if pdf_pages is None or pdf_pages.empty:
        return pd.DataFrame(columns=["document", "page", "evidence_type", "doi", "url", "year_candidates", "snippet"])
    doi_re = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
    url_re = re.compile(r"https?://[^\s<>\]\)]+", re.I)
    year_re = re.compile(r"\b(?:19|20)\d{2}\b")
    rows = []
    for row in pdf_pages.itertuples(index=False):
        text = str(getattr(row, "text", "") or "")
        if not text.strip():
            rows.append({"document": row.document, "page": row.page, "evidence_type": "image-only/empty-text page", "doi": "", "url": "", "year_candidates": "", "snippet": ""})
            continue
        dois = list(dict.fromkeys(m.group(0).rstrip(".,;") for m in doi_re.finditer(text)))
        urls = list(dict.fromkeys(m.group(0).rstrip(".,;") for m in url_re.finditer(text)))
        years = list(dict.fromkeys(year_re.findall(text)))
        # Extractive synopsis: first substantive sentence plus one methods/results-like sentence if available.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.strip()) >= 45]
        candidates = []
        if sentences:
            candidates.append(sentences[0])
            methodish = next((s for s in sentences if re.search(r"\b(method|model|regression|sample|result|finding|estimate|data|analysis|criterion|optim|simulation)\b", s, re.I)), None)
            if methodish and methodish not in candidates:
                candidates.append(methodish)
        snippet = " ".join(candidates)[:1200]
        rows.append({
            "document": row.document,
            "page": int(row.page),
            "evidence_type": "text-extracted page",
            "doi": "; ".join(dois[:5]),
            "url": "; ".join(urls[:5]),
            "year_candidates": "; ".join(years[:10]),
            "snippet": snippet,
        })
    return pd.DataFrame(rows)


def literature_key_terms(pdf_pages: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if pdf_pages is None or pdf_pages.empty:
        return pd.DataFrame(columns=["term", "count"])
    stop = {
        "that","this","with","from","were","have","has","had","which","their","there","these","those","into","between","using","used","than","also","such","more","most","through","within","where","when","what","would","could","should","study","paper","results","result","data","analysis","model","models","method","methods","table","figure","et","al","and","the","for","are","was","not","but","can","may","our","we","of","to","in","a","an","on","as","by","is","be","or","at","it"
    }
    tokens = []
    for text in pdf_pages.text.fillna("").astype(str):
        tokens.extend(re.findall(r"\b[A-Za-z][A-Za-z\-]{3,}\b", text.lower()))
    counts = Counter(t for t in tokens if t not in stop and not t.isdigit())
    return pd.DataFrame(counts.most_common(int(top_n)), columns=["term", "count"])


def generate_research_questions(df: pd.DataFrame, pdf_pages: pd.DataFrame | None = None, limit: int = 150) -> pd.DataFrame:
    """Generate a broad, deterministic question bank from actual columns and literature terms."""
    limit = min(max(1, int(limit)), 200)
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = [c for c in df.columns if c not in numeric and df[c].nunique(dropna=True) <= 80]
    time_cols = [c for c in df.columns if re.search(r"year|date|time|έτος|χρον", str(c), re.I)]
    geo_cols = [c for c in df.columns if re.search(r"region|nuts|municip|geograph|περιφέρ|δήμ", str(c), re.I)]
    terms = literature_key_terms(pdf_pages, top_n=12).term.tolist() if pdf_pages is not None else []
    rows: list[dict[str, Any]] = []
    seen = set()

    def add(question: str, family: str, variables: Sequence[str], priority: int = 2):
        q = re.sub(r"\s+", " ", question).strip()
        if q and q not in seen and len(rows) < limit:
            seen.add(q)
            rows.append({"rq_id": f"RQ{len(rows)+1:03d}", "research_question": q, "method_family": family, "variables": "; ".join(map(str, variables)), "priority": priority})

    # Literature-only operation when no spreadsheet variables are active.
    if not numeric and terms:
        for term in terms:
            add(f"How is '{term}' defined, operationalised and measured across the uploaded literature?", "Literature synthesis", [], 1)
            add(f"What theoretical mechanisms and competing explanations are associated with '{term}' in the uploaded literature?", "Literature synthesis", [], 2)
            add(f"What empirical methods, datasets and identification strategies are used to study '{term}'?", "Methods review", [], 2)
            add(f"What limitations, disagreements and unresolved research gaps surround '{term}'?", "Research gap", [], 2)
            add(f"Which testable hypotheses could be derived from the literature theme '{term}' for a future empirical dataset?", "Hypothesis development", [], 3)
    # Data quality and descriptive questions.
    for v in numeric[:40]:
        add(f"What is the distribution, dispersion, missingness and outlier structure of {v}?", "Descriptive / data quality", [v], 1)
    for a, b in combinations(numeric[:28], 2):
        add(f"How strongly is {a} associated with {b}, and is the relationship robust to rank-based correlation?", "Association / correlation", [a, b], 1)
        if len(rows) >= limit:
            break
    # Group heterogeneity.
    for g in categorical[:12]:
        for y in numeric[:12]:
            add(f"Does {y} differ materially across categories of {g}, and what is the effect size?", "Group comparison", [y, g], 2)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    # Predictive / explanatory questions.
    for y in numeric[:12]:
        preds = [x for x in numeric[:18] if x != y][:5]
        if preds:
            add(f"How accurately can {y} be predicted from {', '.join(preds)}, and which predictors contribute most?", "Predictive + explainable ML", [y, *preds], 2)
            add(f"What is the posterior uncertainty around the association of {', '.join(preds[:3])} with {y}?", "Bayesian modelling", [y, *preds[:3]], 2)
    # Time and geography.
    for t in time_cols[:5]:
        for y in numeric[:15]:
            add(f"How has {y} changed over {t}, and is the observed pattern stable across plausible specifications?", "Longitudinal / time series", [y, t], 2)
    for g in geo_cols[:5]:
        for y in numeric[:15]:
            add(f"How is {y} distributed across {g}, and is there evidence of spatial concentration or regional heterogeneity?", "Spatial / GIS", [y, g], 2)
    # Causal candidates are explicitly framed as design questions rather than causal claims.
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    for t in binary[:8]:
        for y in numeric[:10]:
            if t != y:
                add(f"Under a defensible no-unmeasured-confounding design, what is the estimated average treatment effect of {t} on {y}, and is covariate overlap adequate?", "Causal inference candidate", [t, y], 3)
    # MCDA / Pareto / optimisation candidates.
    if len(numeric) >= 3:
        for a, b in combinations(numeric[:10], 2):
            add(f"Which portfolio choices are Pareto-efficient when simultaneously maximising {a} and {b} under the available resource constraints?", "Pareto / robust optimisation", [a, b], 2)
            if len(rows) >= limit:
                break
    # Literature-connected questions; source terms are merely leads, not interpreted findings.
    for term in terms:
        for y in numeric[:8]:
            add(f"How can the empirical behaviour of {y} be evaluated in relation to the literature theme '{term}' identified in the uploaded PDFs?", "Literature-linked empirical question", [y], 3)
    # Fill literature-only banks with cross-theme questions when no numeric dataset is active.
    if not numeric and terms:
        for a, b in combinations(terms, 2):
            add(f"How are the literature themes '{a}' and '{b}' connected, distinguished or jointly modelled across the uploaded sources?", "Cross-theme literature synthesis", [], 3)
            if len(rows) >= limit:
                break
        cycle = 0
        while len(rows) < limit:
            term = terms[cycle % len(terms)]
            add(f"What additional empirical data, robustness checks and falsification tests would be required to evaluate claims concerning '{term}'?", "Literature-to-empirical design", [], 3)
            add(f"How could competing operational definitions of '{term}' alter the interpretation of future empirical results?", "Measurement robustness", [], 3)
            cycle += 1
            if cycle > limit * 3:
                break
    # Fill to requested size with multivariate robustness questions if necessary.
    cycle = 0
    while len(rows) < limit and numeric:
        y = numeric[cycle % len(numeric)]
        candidates = [x for x in numeric if x != y]
        x = candidates[cycle % len(candidates)] if candidates else y
        add(f"How sensitive is the estimated relationship between {x} and {y} to alternative uncertainty, robust-estimation and subgroup specifications?", "Robustness / sensitivity", [x, y], 3)
        cycle += 1
        if cycle > limit * 10:
            break
    return pd.DataFrame(rows)


def build_agentic_plan(
    df: pd.DataFrame,
    *,
    goal: str,
    outcome: str | None,
    predictors: Sequence[str],
    group: str | None = None,
    time_column: str | None = None,
    region: str | None = None,
    pdf_pages: pd.DataFrame | None = None,
) -> AgenticPlan:
    predictors = [p for p in predictors if p in df.columns and p != outcome]
    steps = []
    if not df.empty:
        steps.append({"step": str(len(steps)+1), "action": "Audit active data", "engine": "deterministic", "output": "quality and missingness tables"})
    if pdf_pages is not None and not pdf_pages.empty:
        steps.append({"step": str(len(steps)+1), "action": "Index uploaded literature PDFs", "engine": "local PDF extraction", "output": "page-level evidence/source index"})
    steps.append({"step": str(len(steps)+1), "action": "Generate and rank research questions", "engine": "deterministic question generator", "output": "up to 150 RQs"})
    if not df.empty:
        steps.append({"step": str(len(steps)+1), "action": "Run descriptive statistics and correlation screen", "engine": "pandas/scipy", "output": "descriptive and association tables"})
    if outcome and predictors:
        steps.append({"step": "5", "action": f"Estimate primary OLS model for {outcome}", "engine": "statsmodels HC3", "output": "coefficients, diagnostics and fitted values"})
    if group and outcome:
        steps.append({"step": "6", "action": f"Test heterogeneity of {outcome} across {group}", "engine": "Welch/Mann-Whitney or ANOVA/Kruskal", "output": "tests + effect sizes"})
    if len(predictors) >= 2:
        steps.append({"step": "7", "action": "Run PCA and unsupervised segmentation", "engine": "scikit-learn", "output": "loadings, explained variance and cluster profiles"})
    if time_column:
        steps.append({"step": "8", "action": f"Preserve {time_column} as temporal routing information", "engine": "method router", "output": "time-aware recommendations"})
    if region:
        steps.append({"step": "9", "action": f"Preserve {region} for GIS/spatial routing", "engine": "method router", "output": "spatial-analysis recommendation"})
    steps.extend([
        {"step": str(len(steps)+1), "action": "Synthesize evidence-grounded discussion and conclusions", "engine": "offline deterministic narrative", "output": "draft discussion + conclusion"},
        {"step": str(len(steps)+2), "action": "Build submission package", "engine": "local export system", "output": "DOCX + XLSX + JSON + interactive HTML + publication graphics"},
    ])
    warnings = [
        "The agent never infers causality from an observational table without an explicit causal design.",
        "Extracted PDF text is research evidence, not automatically verified bibliographic metadata or quotation-ready text.",
        "The package is a near-submission draft and still requires human verification of claims, references, formatting and institutional requirements.",
        "External LLM use is optional; numerical analysis and the offline package remain available without an API key.",
    ]
    mappings = {"outcome": outcome, "predictors": list(predictors), "group": group, "time_column": time_column, "region": region, "pdf_documents": sorted(pdf_pages.document.unique().tolist()) if pdf_pages is not None and not pdf_pages.empty else []}
    return AgenticPlan(goal=goal.strip(), steps=steps, warnings=warnings, mappings=mappings)


def _top_correlations(corr: pd.DataFrame, pvals: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            r = corr.loc[a, b]
            p = pvals.loc[a, b]
            if pd.notna(r):
                rows.append({"variable_1": a, "variable_2": b, "correlation": float(r), "abs_correlation": abs(float(r)), "p_value": float(p) if pd.notna(p) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).head(n) if rows else pd.DataFrame(columns=["variable_1","variable_2","correlation","abs_correlation","p_value"])


def _correlation_figure_bytes(corr: pd.DataFrame) -> tuple[dict[str, bytes], dict[str, bytes]]:
    if corr.empty:
        return {}, {}
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Correlation matrix", loc="left", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.035, pad=.02)
    static = {}
    for fmt in ["png", "svg", "pdf"]:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight", facecolor="white")
        static[f"correlation_matrix.{fmt}"] = buf.getvalue()
    plt.close(fig)
    try:
        import plotly.express as px
        pfig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Interactive correlation matrix", aspect="auto")
        interactive = {"correlation_matrix.html": pfig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")}
    except Exception:
        interactive = {}
    return static, interactive


def _ols_figure_bytes(coef: pd.DataFrame) -> tuple[dict[str, bytes], dict[str, bytes]]:
    if coef is None or coef.empty or not {"term", "coefficient", "ci_95_low", "ci_95_high"}.issubset(coef.columns):
        return {}, {}
    d = coef[~coef.term.astype(str).str.lower().isin(["const", "intercept"])].copy().head(30)
    if d.empty:
        return {}, {}
    d = d.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, max(4, .36 * len(d) + 1.5)), constrained_layout=True)
    y = np.arange(len(d))
    ax.errorbar(d.coefficient, y, xerr=[d.coefficient - d.ci_95_low, d.ci_95_high - d.coefficient], fmt="o", capsize=3)
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(d.term, fontsize=8)
    ax.set_xlabel("Coefficient (95% CI)"); ax.set_title("Primary OLS coefficient estimates", loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    static = {}
    for fmt in ["png", "svg", "pdf"]:
        buf = io.BytesIO(); fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight", facecolor="white"); static[f"ols_coefficients.{fmt}"] = buf.getvalue()
    plt.close(fig)
    try:
        import plotly.express as px
        pfig = px.scatter(d, x="coefficient", y="term", error_x=d["ci_95_high"]-d["coefficient"], error_x_minus=d["coefficient"]-d["ci_95_low"], title="Primary OLS coefficient estimates")
        pfig.add_vline(x=0, line_dash="dash")
        interactive = {"ols_coefficients.html": pfig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")}
    except Exception:
        interactive = {}
    return static, interactive


def offline_agent_reply(run: AgenticRun, question: str) -> str:
    """Deterministic conversation over computed tables; no external AI required."""
    q = str(question).strip().casefold()
    if not q:
        return "Ask about the strongest finding, conclusions, limitations, literature, research questions, model results or next analyses."
    if any(k in q for k in ["strongest", "important", "main finding", "key finding", "σημαν"]):
        top = run.tables.get("Top correlations", pd.DataFrame())
        if not top.empty:
            r = top.iloc[0]
            return f"The strongest screened pairwise association is {r.variable_1} ↔ {r.variable_2}: r={r.correlation:.3f}, p={r.p_value:.4g}. This is an association screen, not a causal estimate. The exported OLS/diagnostic tables should be used to decide whether it survives the configured multivariable specification."
        return run.narratives.get("discussion", "No ranked association table is available for this run.")
    if any(k in q for k in ["conclusion", "conclude", "συμπέρα", "συμπερα"]):
        return run.narratives.get("conclusion", "No conclusion draft is available.")
    if any(k in q for k in ["limit", "weak", "caveat", "περιορισ"]):
        return run.narratives.get("limitations", "No limitations draft is available.")
    if any(k in q for k in ["literature", "source", "pdf", "bibliograph", "βιβλιο", "πηγ"]):
        terms = run.tables.get("Literature key terms", pd.DataFrame())
        ev = run.tables.get("Literature source evidence", pd.DataFrame())
        if ev.empty:
            return "No literature PDFs were indexed in this run."
        topic_text = ", ".join(terms.term.head(10).astype(str)) if not terms.empty else "no stable key-term list"
        doi_count = int(ev.doi.astype(str).str.len().gt(0).sum()) if "doi" in ev else 0
        return f"The offline literature pass indexed {ev.document.nunique()} document(s) and {len(ev)} page-level evidence records. Frequent terms include {topic_text}. DOI strings were detected on {doi_count} page records. These are navigation/evidence notes; verify bibliographic metadata and quotations in the originals before submission."
    if any(k in q for k in ["research question", "rq", "ερευνητικ"]):
        rqs = run.tables.get("Research questions", pd.DataFrame())
        if rqs.empty:
            return "No research-question bank was generated."
        top = rqs.head(10)
        return "Top generated questions:\n" + "\n".join(f"{r.rq_id}. {r.research_question}" for r in top.itertuples(index=False))
    if any(k in q for k in ["model", "regression", "ols", "coefficient", "παλινδ"]):
        coef = run.tables.get("OLS coefficients", pd.DataFrame())
        fit = run.tables.get("OLS fit", pd.DataFrame())
        if coef.empty:
            return "No primary OLS model was executed. Map a primary outcome and predictors, rebuild the plan and approve the run."
        nonconst = coef[~coef.term.astype(str).str.lower().isin(["const", "intercept"])].copy()
        nonconst["p_value"] = pd.to_numeric(nonconst.p_value, errors="coerce")
        nonconst = nonconst.sort_values("p_value").head(5)
        fit_text = fit.head(1).to_dict("records")[0] if not fit.empty else {}
        terms = "; ".join(f"{r.term}: β={r.coefficient:.4g}, p={r.p_value:.4g}" for r in nonconst.itertuples(index=False))
        return f"Primary model summary: {fit_text}. Lowest-p-value non-intercept terms: {terms}. Interpret effect sizes and diagnostics, not p-values alone."
    if any(k in q for k in ["next", "further", "additional", "what else", "επόμεν", "περαιτέρω"]):
        suggestions = ["Re-check missingness and influential observations before final inference.", "Use the Frontier Methods Lab when the question is explicitly causal, Bayesian, Pareto/multi-objective or explainability-focused.", "Run subgroup/temporal/spatial robustness only when the mapped variables represent those dimensions validly.", "Re-run the approved Agentic workflow after any data, variable or model change so the manifest matches the manuscript."]
        return "Recommended next steps:\n- " + "\n- ".join(suggestions)
    return ("The offline agent is intentionally bounded. I can answer from the computed run about the strongest findings, model coefficients, literature evidence, generated research questions, conclusions, limitations and next analyses. "
            "For open-ended synthesis, enable the optional LLM mode; numerical evidence remains unchanged.")


def _offline_narrative(tables: dict[str, pd.DataFrame], plan: AgenticPlan) -> dict[str, str]:
    desc = tables.get("Descriptive statistics", pd.DataFrame())
    top = tables.get("Top correlations", pd.DataFrame())
    coef = tables.get("OLS coefficients", pd.DataFrame())
    quality = tables.get("Quality audit", pd.DataFrame())
    paragraphs = []
    if not quality.empty:
        pairs = "; ".join(f"{r.check}: {r.value}" for r in quality.head(6).itertuples(index=False))
        paragraphs.append(f"The automated audit characterised the active analytical scope before modelling ({pairs}).")
    lit = tables.get("Literature source evidence", pd.DataFrame())
    terms = tables.get("Literature key terms", pd.DataFrame())
    if not lit.empty:
        docs = int(lit.document.nunique()) if "document" in lit else 0
        pages = len(lit)
        topic_text = ", ".join(terms.term.head(8).astype(str)) if not terms.empty else "no stable key terms extracted"
        paragraphs.append(f"The local literature pass indexed {docs} document(s) across {pages} extracted pages. Frequent source terms included {topic_text}. These are extractive navigation signals rather than verified claims or complete bibliographic records.")
    if not desc.empty:
        most_variable = desc.assign(abs_cv=pd.to_numeric(desc.cv, errors="coerce").abs()).sort_values("abs_cv", ascending=False).head(1)
        if not most_variable.empty and pd.notna(most_variable.iloc[0].abs_cv):
            r = most_variable.iloc[0]
            paragraphs.append(f"Among the screened numeric variables, {r.variable} showed the greatest relative dispersion (|CV|={r.abs_cv:.3f}); distributional shape and outliers should therefore be reported alongside the mean.")
    if not top.empty:
        r = top.iloc[0]
        paragraphs.append(f"The strongest screened pairwise association was between {r.variable_1} and {r.variable_2} (r={r.correlation:.3f}, p={r.p_value:.4g}). This is an association screen, not a causal estimate.")
    if not coef.empty:
        nonconst = coef[~coef.term.astype(str).str.lower().isin(["const", "intercept"])]
        if not nonconst.empty:
            r = nonconst.iloc[np.nanargmin(pd.to_numeric(nonconst.p_value, errors="coerce").fillna(1).to_numpy())]
            paragraphs.append(f"In the configured primary OLS model, the lowest p-value among non-intercept terms corresponded to {r.term} (β={r.coefficient:.4g}, 95% CI {r.ci_95_low:.4g} to {r.ci_95_high:.4g}, p={r.p_value:.4g}). Interpretation remains conditional on model specification and diagnostics.")
    if not paragraphs:
        paragraphs.append("The deterministic workflow completed the available audit and evidence-indexing steps. Additional model interpretation requires explicit variable mappings or a richer analytical scope.")
    discussion = "\n\n".join(paragraphs)
    conclusion = (f"For the research goal — {plan.goal or 'the configured research objective'} — the automated workflow provides a reproducible first-pass empirical synthesis. "
                  "The strongest defensible conclusions are those directly supported by the exported tables and diagnostics. Causal language, exact bibliographic claims and final submission formatting require explicit verification.")
    limitations = "The automated package is designed to remove repetitive analytical work, not scholarly responsibility. Missing data, non-random sampling, measurement validity, model specification, multiple testing, external validity and causal identification must be reviewed before submission."
    return {"discussion": discussion, "conclusion": conclusion, "limitations": limitations}


def run_agentic_workflow(
    df: pd.DataFrame,
    *,
    plan: AgenticPlan,
    pdf_pages: pd.DataFrame | None = None,
    outcome: str | None = None,
    predictors: Sequence[str] = (),
    group: str | None = None,
    question_limit: int = 150,
) -> AgenticRun:
    tables: dict[str, pd.DataFrame] = {}
    tables["Quality audit"] = quality_summary(df)
    numeric = list(df.select_dtypes(include=np.number).columns)
    selected_numeric = list(dict.fromkeys([*(predictors or []), *([outcome] if outcome else [])]))
    selected_numeric = [c for c in selected_numeric if c in numeric]
    if len(selected_numeric) < 2:
        selected_numeric = numeric[:min(30, len(numeric))]
    tables["Descriptive statistics"] = descriptive_statistics(df, selected_numeric or numeric[:30])
    if selected_numeric:
        corr, pvals = correlation_matrix(df, selected_numeric, method="pearson")
        tables["Correlation matrix"] = corr.reset_index(names="variable")
        tables["Correlation p-values"] = pvals.reset_index(names="variable")
        tables["Top correlations"] = _top_correlations(corr, pvals)
        static, interactive = _correlation_figure_bytes(corr)
    else:
        corr = pd.DataFrame(); static, interactive = {}, {}
    ols_static, ols_interactive = {}, {}
    if outcome and predictors:
        try:
            model = fit_detailed_model(df, outcome, list(predictors), estimator="OLS", covariance="HC3")
            tables["OLS coefficients"] = model.coefficients
            tables["OLS fit"] = model.fit
            tables["OLS diagnostics"] = model.diagnostics
            tables["OLS predictions"] = model.predictions
            ols_static, ols_interactive = _ols_figure_bytes(model.coefficients)
        except Exception as exc:
            ols_static, ols_interactive = {}, {}
            tables["OLS execution note"] = pd.DataFrame([{"status": "not executed", "reason": str(exc)}])
    if group and outcome:
        try:
            tables["Group tests"] = group_tests(df, [outcome], group)
            tables["Group frequencies"] = categorical_summary(df, [group])
        except Exception as exc:
            tables["Group execution note"] = pd.DataFrame([{"status": "not executed", "reason": str(exc)}])
    if len(selected_numeric) >= 2 and len(df) >= 10:
        try:
            loadings, variance = pca_table(df, selected_numeric[:20], n_components=min(5, len(selected_numeric)))
            tables["PCA loadings"] = loadings
            tables["PCA variance"] = variance
        except Exception as exc:
            tables["PCA execution note"] = pd.DataFrame([{"status": "not executed", "reason": str(exc)}])
        try:
            assignments, profiles = cluster_table(df, selected_numeric[:12], clusters=min(4, max(2, len(df)//20)))
            tables["Cluster assignments"] = assignments
            tables["Cluster profiles"] = profiles
        except Exception as exc:
            tables["Cluster execution note"] = pd.DataFrame([{"status": "not executed", "reason": str(exc)}])
    source_evidence = extract_source_evidence(pdf_pages if pdf_pages is not None else pd.DataFrame())
    tables["Literature source evidence"] = source_evidence
    tables["Literature key terms"] = literature_key_terms(pdf_pages if pdf_pages is not None else pd.DataFrame())
    tables["Research questions"] = generate_research_questions(df, pdf_pages, limit=question_limit)
    static.update(ols_static)
    interactive.update(ols_interactive)
    narratives = _offline_narrative(tables, plan)
    manifest = {
        "version": "5.8.0",
        "mode": "offline deterministic agentic workflow",
        "goal": plan.goal,
        "mappings": plan.mappings,
        "steps": plan.steps,
        "warnings": plan.warnings,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "tables_generated": sorted(tables),
        "research_questions_generated": int(len(tables["Research questions"])),
        "pdf_documents": sorted(pdf_pages.document.unique().tolist()) if pdf_pages is not None and not pdf_pages.empty else [],
    }
    return AgenticRun(plan=plan, tables=tables, narratives=narratives, figures=static, interactive_html=interactive, manifest=manifest)


def _paper_markdown(run: AgenticRun, title: str = "Agentic Research Draft") -> str:
    q = run.tables.get("Research questions", pd.DataFrame()).head(10)
    q_text = "\n".join(f"- {r.research_question}" for r in q.itertuples(index=False)) or "- No research questions generated."
    source = run.tables.get("Literature source evidence", pd.DataFrame())
    source_lines = []
    if not source.empty:
        for row in source.head(60).itertuples(index=False):
            doi = f" DOI: {row.doi}." if getattr(row, "doi", "") else ""
            source_lines.append(f"- {row.document}, p. {row.page}.{doi} {row.snippet[:300]}")
    sources = "\n".join(source_lines) or "- No PDF evidence supplied."
    return f"""# {title}

## Status

Near-submission analytical draft generated by Makryvelios Agentic Research Mode. Human verification of claims, references, formatting and institutional requirements remains mandatory.

## Research objective

{run.plan.goal or '[Define the principal research objective]'}

## Candidate research questions

{q_text}

## Data and methods

The workflow audited the active dataset, produced descriptive and association evidence, and executed only analytical routines supported by the configured variable mappings. Where a primary outcome and predictors were supplied, HC3-robust OLS was estimated; PCA and clustering were run where the numeric matrix was sufficient. All tables, diagnostics, mappings and workflow steps are included in the reproducibility package.

## Results

{run.narratives.get('discussion','')}

## Discussion

The reported patterns should be interpreted in relation to the substantive research design and uploaded literature. Automated synthesis prioritises reproducible numerical evidence and explicitly separates association, prediction, optimisation and causal claims.

## Conclusion

{run.narratives.get('conclusion','')}

## Limitations

{run.narratives.get('limitations','')}

## Literature evidence notes

{sources}

## Submission checklist

1. Verify every bibliographic reference and quotation against the original PDF.
2. Confirm the final research question, hypotheses and causal identification language.
3. Review model diagnostics, missing-data handling and multiplicity.
4. Apply the institution/journal style guide and final reference manager output.
5. Re-run the workflow after any data or specification change and retain the exported manifest.
"""


def _agentic_docx_bytes(run: AgenticRun, title: str) -> bytes:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError as exc:
        raise RuntimeError("Word export requires python-docx from the bundled requirements.") from exc
    document = Document()
    styles = document.styles
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2", "Heading 3"]:
        if style_name in styles:
            styles[style_name].font.name = "Times New Roman"
            if style_name == "Normal":
                styles[style_name].font.size = Pt(11)
    document.add_heading(title, 0)
    p = document.add_paragraph()
    r = p.add_run("Near-submission analytical draft generated by Makryvelios Agentic Research Mode. Human verification is required before submission.")
    r.bold = True
    document.add_heading("Research objective", level=1)
    document.add_paragraph(run.plan.goal or "[Define the principal research objective]")
    document.add_heading("Candidate research questions", level=1)
    rqs = run.tables.get("Research questions", pd.DataFrame()).head(15)
    if rqs.empty:
        document.add_paragraph("No research questions were generated.")
    else:
        for row in rqs.itertuples(index=False):
            document.add_paragraph(f"{row.rq_id}. {row.research_question}", style="List Number")
    document.add_heading("Data and methods", level=1)
    document.add_paragraph("The approved workflow audited the active dataset, indexed selected literature evidence locally, generated a candidate research-question bank and executed only analytical routines supported by the mapped variables. Numerical outputs are deterministic and do not require an external LLM API.")
    plan_table = document.add_table(rows=1, cols=4)
    plan_table.style = "Table Grid"
    for i, h in enumerate(["Step", "Action", "Engine", "Output"]): plan_table.rows[0].cells[i].text = h
    for item in run.plan.steps:
        cells = plan_table.add_row().cells
        for i, key in enumerate(["step", "action", "engine", "output"]): cells[i].text = str(item.get(key, ""))
    document.add_heading("Results", level=1)
    document.add_paragraph(run.narratives.get("discussion", ""))
    for table_name in ["OLS coefficients", "OLS fit", "OLS diagnostics", "Top correlations", "Group tests", "PCA variance"]:
        table = run.tables.get(table_name, pd.DataFrame())
        if table is None or table.empty:
            continue
        document.add_heading(table_name, level=2)
        clipped = table.head(25).copy()
        doc_table = document.add_table(rows=1, cols=len(clipped.columns))
        doc_table.style = "Table Grid"
        for j, col in enumerate(clipped.columns): doc_table.rows[0].cells[j].text = str(col)
        for _, row in clipped.iterrows():
            cells = doc_table.add_row().cells
            for j, col in enumerate(clipped.columns):
                value = row[col]
                cells[j].text = "" if pd.isna(value) else (f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value))
    png = run.figures.get("correlation_matrix.png") or run.figures.get("ols_coefficients.png")
    if png:
        document.add_heading("Selected analytical figure", level=2)
        document.add_picture(io.BytesIO(png), width=Inches(6.3))
    document.add_heading("Discussion", level=1)
    document.add_paragraph(run.narratives.get("discussion", ""))
    document.add_heading("Conclusion", level=1)
    document.add_paragraph(run.narratives.get("conclusion", ""))
    document.add_heading("Limitations", level=1)
    document.add_paragraph(run.narratives.get("limitations", ""))
    document.add_heading("Literature evidence notes", level=1)
    evidence = run.tables.get("Literature source evidence", pd.DataFrame()).head(40)
    if evidence.empty:
        document.add_paragraph("No PDF evidence was supplied.")
    else:
        for row in evidence.itertuples(index=False):
            doi = f" DOI: {row.doi}." if getattr(row, "doi", "") else ""
            document.add_paragraph(f"{row.document}, p. {row.page}.{doi} {str(row.snippet)[:450]}", style="List Bullet")
    document.add_heading("Submission verification checklist", level=1)
    for item in [
        "Verify every bibliographic reference and quotation against the original PDF.",
        "Confirm the final research question, hypotheses and any causal-identification language.",
        "Review model diagnostics, missing-data handling, multiplicity and robustness.",
        "Apply the institution/journal style guide and final reference-manager output.",
        "Re-run after any data/specification change and retain the matching manifest.",
    ]:
        document.add_paragraph(item, style="List Bullet")
    out = io.BytesIO(); document.save(out); return out.getvalue()


def _html_report(run: AgenticRun, title: str) -> bytes:
    sections = [f"<h1>{html.escape(title)}</h1>", "<p><b>Near-submission analytical draft; human verification required.</b></p>"]
    sections.append(f"<h2>Research objective</h2><p>{html.escape(run.plan.goal)}</p>")
    sections.append("<h2>Discussion</h2><p>" + html.escape(run.narratives.get("discussion", "")).replace("\n", "<br>") + "</p>")
    sections.append("<h2>Conclusion</h2><p>" + html.escape(run.narratives.get("conclusion", "")).replace("\n", "<br>") + "</p>")
    for name, table in run.tables.items():
        if table is None or table.empty:
            continue
        sections.append(f"<h2>{html.escape(name)}</h2>")
        sections.append(table.head(500).to_html(index=False, escape=True))
    for name, payload in run.interactive_html.items():
        # Include linkable self-contained interactive files in ZIP rather than nesting huge JS here.
        sections.append(f"<p>Interactive figure included in package: <code>interactive/{html.escape(name)}</code></p>")
    css = "body{font-family:Georgia,serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ccd5df;padding:6px;vertical-align:top}th{background:#e9eef4}h1,h2{color:#173a63}code{background:#f1f5f9;padding:2px 4px}"
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + html.escape(title) + "</title><style>" + css + "</style></head><body>" + "\n".join(sections) + "</body></html>").encode("utf-8")


def agentic_submission_package(run: AgenticRun, title: str = "Agentic Research Draft") -> bytes:
    """Build a complete near-submission package: DOCX, XLSX, JSON, HTML and figures."""
    paper = _paper_markdown(run, title=title)
    workbook = to_excel_bytes({name[:31]: table for name, table in run.tables.items() if isinstance(table, pd.DataFrame)})
    report = _html_report(run, title)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("paper/paper_draft.docx", _agentic_docx_bytes(run, title))
        archive.writestr("paper/paper_draft.md", paper.encode("utf-8"))
        archive.writestr("results/complete_results.xlsx", workbook)
        archive.writestr("results/manifest.json", json.dumps(run.manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        archive.writestr("report/interactive_report.html", report)
        for name, payload in run.figures.items():
            archive.writestr(f"figures/{name}", payload)
        for name, payload in run.interactive_html.items():
            archive.writestr(f"interactive/{name}", payload)
        archive.writestr("README.txt", (
            "Makryvelios Agentic Research Mode v5.8.0\n\n"
            "This package is generated locally from the configured data and PDF evidence.\n"
            "Numerical outputs do not require an external AI API.\n"
            "The draft is designed to be close to submission-ready but must be checked by a human researcher before submission.\n"
            "Verify references, quotations, causal claims, institutional formatting and every substantive conclusion.\n"
        ).encode("utf-8"))
    return buf.getvalue()
