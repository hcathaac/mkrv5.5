"""Interactive and publication-ready colour/monochrome figures."""
from __future__ import annotations

import io
import zipfile
from typing import Sequence

import numpy as np
import pandas as pd

from output_guidance import build_output_guide, output_guide_markdown
import plotly.express as px
from matplotlib import pyplot as plt


COLOUR = ["#155B8A", "#D89B2B", "#C45A36", "#6F7D3C", "#9A5C83", "#3B7D6E"]
GREYS = ["#111111", "#4D4D4D", "#777777", "#9E9E9E", "#BDBDBD", "#D9D9D9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def interactive_figure(df: pd.DataFrame, chart: str, x: str, y: str, group: str | None = None, title: str | None = None):
    data = df[[c for c in [x, y, group] if c]].dropna().copy()
    title = title or f"{y} by {x}"
    if chart == "Line":
        fig = px.line(data, x=x, y=y, color=group, markers=True, title=title)
    elif chart == "Scatter":
        fig = px.scatter(data, x=x, y=y, color=group, trendline="ols" if not group else None, title=title)
    elif chart == "Box":
        fig = px.box(data, x=x, y=y, color=group, points="outliers", title=title)
    elif chart == "Histogram":
        fig = px.histogram(data, x=y, color=group, marginal="box", title=title)
    else:
        agg = data.groupby([x] + ([group] if group else []), dropna=False)[y].mean().reset_index()
        fig = px.bar(agg, x=x, y=y, color=group, title=title)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,43,60,.78)", font=dict(family="Arial", size=13), legend_title_text=group or "", margin=dict(l=55, r=25, t=70, b=55))
    return fig


def _static_figure(df: pd.DataFrame, chart: str, x: str, y: str, group: str | None, monochrome: bool, title: str):
    cols = [c for c in [x, y, group] if c]
    data = df[cols].dropna().copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[y])
    palette = GREYS if monochrome else COLOUR
    fig, ax = plt.subplots(figsize=(8.27, 5.7), constrained_layout=True)
    groups = [(None, data)] if not group else list(data.groupby(group, dropna=False))
    if chart == "Histogram":
        for i, (label, part) in enumerate(groups):
            ax.hist(part[y], bins="auto", color=palette[i % len(palette)], alpha=.55 if len(groups) > 1 else .82, edgecolor="#222222", linewidth=.5, label=str(label) if label is not None else None, hatch="" if not monochrome else ["", "//", "xx", ".."] [i % 4])
        ax.set_xlabel(y); ax.set_ylabel("Frequency")
    elif chart == "Scatter":
        for i, (label, part) in enumerate(groups):
            xx = pd.to_numeric(part[x], errors="coerce")
            keep = xx.notna()
            ax.scatter(xx[keep], part.loc[keep, y], s=30, facecolors="none" if monochrome and i % 2 else palette[i % len(palette)], edgecolors=palette[i % len(palette)], marker=MARKERS[i % len(MARKERS)], label=str(label) if label is not None else None, alpha=.82)
        ax.set_xlabel(x); ax.set_ylabel(y)
    elif chart == "Line":
        for i, (label, part) in enumerate(groups):
            agg = part.groupby(x, dropna=False)[y].mean().reset_index().sort_values(x)
            ax.plot(agg[x], agg[y], color=palette[i % len(palette)], marker=MARKERS[i % len(MARKERS)], linestyle=["-", "--", "-.", ":"][i % 4], label=str(label) if label is not None else None, linewidth=1.7)
        ax.set_xlabel(x); ax.set_ylabel(y)
    elif chart == "Box":
        categories = list(data[x].astype(str).drop_duplicates())[:30]
        values = [data.loc[data[x].astype(str) == cat, y].dropna() for cat in categories]
        boxes = ax.boxplot(values, labels=categories, patch_artist=True, showfliers=True)
        for i, box in enumerate(boxes["boxes"]):
            box.set(facecolor=palette[i % len(palette)], alpha=.65, edgecolor="#222222", hatch="" if not monochrome else ["", "//", "xx", ".."][i % 4])
        ax.set_xlabel(x); ax.set_ylabel(y)
    else:
        group_cols = [x] + ([group] if group else [])
        agg = data.groupby(group_cols, dropna=False)[y].mean().reset_index()
        if group:
            pivot = agg.pivot(index=x, columns=group, values=y).head(30)
            positions = np.arange(len(pivot)); width = .8 / max(len(pivot.columns), 1)
            for i, col in enumerate(pivot.columns):
                ax.bar(positions - .4 + width / 2 + i * width, pivot[col], width, label=str(col), color=palette[i % len(palette)], edgecolor="#222222", linewidth=.45, hatch="" if not monochrome else ["", "//", "xx", ".."][i % 4])
            ax.set_xticks(positions, pivot.index.astype(str), rotation=45, ha="right")
        else:
            agg = agg.sort_values(y, ascending=False).head(30)
            ax.bar(agg[x].astype(str), agg[y], color=palette[0], edgecolor="#222222", linewidth=.45, hatch="" if not monochrome else "//")
            ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel(x); ax.set_ylabel(f"Mean {y}")
    if group and len(groups) <= 12:
        ax.legend(frameon=False, ncol=min(3, len(groups)), loc="best")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=10)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.6, alpha=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.text(.01, .005, f"Source: uploaded analytical data. N={len(data):,}. Values shown without causal interpretation.", fontsize=7.5, color="#444444")
    return fig


def publication_bundle(df: pd.DataFrame, chart: str, x: str, y: str, group: str | None = None, title: str | None = None) -> bytes:
    title = title or f"{y} by {x}"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            fig = _static_figure(df, chart, x, y, group, monochrome, title)
            for fmt in ("png", "svg", "pdf"):
                buf = io.BytesIO()
                fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight", facecolor="white")
                zf.writestr(f"{style}_{chart.lower()}.{fmt}", buf.getvalue())
            plt.close(fig)
        data_cols = [c for c in [x, y, group] if c]
        figure_data = df[data_cols]
        zf.writestr("figure_data.csv", figure_data.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("figure_notes.txt", f"Title: {title}\nChart: {chart}\nX: {x}\nY: {y}\nGroup: {group or 'None'}\nRaster resolution: 600 dpi\nVector formats: SVG and PDF\n")
        _write_output_guides(zf, {"Figure data": figure_data}, [f"{title} ({chart} chart)"])
    return archive.getvalue()


def _save_figure_set(zf: zipfile.ZipFile, fig, stem: str) -> None:
    """Write one figure as 600-dpi PNG and vector SVG/PDF."""
    for fmt in ("png", "svg", "pdf"):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight", facecolor="white")
        zf.writestr(f"{stem}.{fmt}", buf.getvalue())


def _write_output_guides(
    zf: zipfile.ZipFile,
    tables: dict[str, pd.DataFrame],
    charts: list[str],
) -> None:
    guide = build_output_guide(tables, charts)
    zf.writestr("output_interpretation_guide.csv", guide.to_csv(index=False).encode("utf-8-sig"))
    zf.writestr("OUTPUT_INTERPRETATION_GUIDE.md", output_guide_markdown(guide))


def ols_publication_bundle(predictions: pd.DataFrame, coefficients: pd.DataFrame, title: str = "OLS diagnostics") -> bytes:
    """Publication package for OLS fit, residual and coefficient evidence."""
    archive = io.BytesIO()
    pred = predictions.copy()
    coef = coefficients[coefficients.term != "const"].copy().sort_values("coefficient")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            main = GREYS[1] if monochrome else COLOUR[0]
            accent = GREYS[0] if monochrome else COLOUR[1]
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4), constrained_layout=True)
            axes[0].scatter(pred.observed, pred.fitted, s=18, facecolors="none", edgecolors=main, alpha=.72)
            lo = float(np.nanmin([pred.observed.min(), pred.fitted.min()])); hi = float(np.nanmax([pred.observed.max(), pred.fitted.max()]))
            axes[0].plot([lo, hi], [lo, hi], color=accent, linestyle="--", linewidth=1.5)
            axes[0].set(xlabel="Observed", ylabel="Fitted", title="Observed versus fitted")
            axes[1].scatter(pred.fitted, pred.residual, s=18, facecolors="none", edgecolors=main, alpha=.72)
            axes[1].axhline(0, color=accent, linestyle="--", linewidth=1.5)
            axes[1].set(xlabel="Fitted", ylabel="Residual", title="Residuals versus fitted")
            for ax in axes:
                ax.grid(axis="both", color="#D9D9D9", linewidth=.55, alpha=.7); ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(title, x=.01, ha="left", fontsize=15, fontweight="bold")
            _save_figure_set(zf, fig, f"{style}_ols_diagnostics")
            plt.close(fig)
            if not coef.empty:
                shown = coef.tail(40)
                fig, ax = plt.subplots(figsize=(8.27, max(4.8, .25 * len(shown))), constrained_layout=True)
                ypos = np.arange(len(shown))
                ax.errorbar(shown.coefficient, ypos,
                            xerr=[shown.coefficient - shown.ci_95_low, shown.ci_95_high - shown.coefficient],
                            fmt="o", color=main, ecolor=accent, capsize=2.5)
                ax.axvline(0, color="#333333", linewidth=.9, linestyle="--")
                ax.set_yticks(ypos, shown.term.astype(str)); ax.set_xlabel("Coefficient and 95% confidence interval")
                ax.set_title("OLS coefficient plot", loc="left", fontsize=15, fontweight="bold")
                ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
                _save_figure_set(zf, fig, f"{style}_ols_coefficients")
                plt.close(fig)
        zf.writestr("ols_predictions.csv", pred.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("ols_coefficients.csv", coefficients.to_csv(index=False).encode("utf-8-sig"))
        _write_output_guides(
            zf,
            {"OLS predictions and residuals": pred, "OLS coefficients": coefficients},
            ["Observed versus fitted chart", "Residuals versus fitted chart", "OLS coefficient chart"],
        )
        zf.writestr("README.txt", "OLS publication figures. Raster files are 600 dpi; SVG and PDF are vector formats. Colour and black-and-white variants are included.\n")
    return archive.getvalue()


def monte_carlo_publication_bundle(draws: pd.DataFrame, summary: pd.DataFrame, term: str, title: str | None = None) -> bytes:
    """Colour and monochrome publication figures for a Monte Carlo coefficient."""
    if term not in draws:
        raise ValueError(f"Simulation term not found: {term}")
    values = pd.to_numeric(draws[term], errors="coerce").dropna()
    title = title or f"Monte Carlo distribution: {term}"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            colour = GREYS[1] if monochrome else COLOUR[0]
            accent = GREYS[0] if monochrome else COLOUR[1]
            fig, ax = plt.subplots(figsize=(8.27, 5.7), constrained_layout=True)
            ax.hist(values, bins="auto", density=True, color=colour, alpha=.78, edgecolor="white", linewidth=.45,
                    hatch="//" if monochrome else "")
            ax.axvline(values.median(), color=accent, linewidth=1.8, linestyle="--", label="Simulation median")
            ax.axvline(0, color="#333333", linewidth=1.0, linestyle=":", label="Zero")
            ax.set(xlabel="Simulated coefficient", ylabel="Density")
            ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
            ax.legend(frameon=False); ax.grid(axis="y", color="#D9D9D9", linewidth=.55)
            ax.spines[["top", "right"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_monte_carlo_{term}")
            plt.close(fig)
        zf.writestr("simulation_draws.csv", draws.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("simulation_summary.csv", summary.to_csv(index=False).encode("utf-8-sig"))
        _write_output_guides(
            zf,
            {"Monte Carlo simulation draws": draws, "Monte Carlo simulation summary": summary},
            [f"Monte Carlo coefficient distribution: {term}"],
        )
        zf.writestr("README.txt", f"Monte Carlo coefficient: {term}. Raster files are 600 dpi; SVG and PDF are vector formats.\n")
    return archive.getvalue()


def clustering_publication_bundle(assignments: pd.DataFrame, profiles: pd.DataFrame, embedding: pd.DataFrame, diagnostics: pd.DataFrame) -> bytes:
    """Publication-ready cluster projection and profile figures."""
    archive = io.BytesIO()
    clusters = sorted(embedding.cluster.unique())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            palette = GREYS if monochrome else COLOUR
            fig, ax = plt.subplots(figsize=(8.27, 5.7), constrained_layout=True)
            for i, cluster in enumerate(clusters):
                part = embedding[embedding.cluster == cluster]
                label = "Noise / outlier" if cluster == -1 else f"Cluster {cluster}"
                ax.scatter(part.dimension_1, part.dimension_2, s=28, alpha=.72,
                           color=palette[i % len(palette)], marker=MARKERS[i % len(MARKERS)],
                           facecolors="none" if monochrome and i % 2 else palette[i % len(palette)],
                           edgecolors=palette[i % len(palette)], label=label)
            ax.set(xlabel="Dimension 1", ylabel="Dimension 2")
            ax.set_title("Cluster projection", loc="left", fontsize=15, fontweight="bold")
            ax.legend(frameon=False, ncol=min(3, len(clusters))); ax.grid(color="#D9D9D9", linewidth=.55)
            ax.spines[["top", "right"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_cluster_projection")
            plt.close(fig)

            mean_cols = [c for c in profiles if c.startswith("mean_")][:30]
            if mean_cols:
                matrix = profiles[mean_cols].to_numpy(float)
                centre = np.nanmean(matrix, axis=0); spread = np.nanstd(matrix, axis=0)
                z = np.divide(matrix - centre, spread, out=np.zeros_like(matrix), where=spread > 0)
                fig, ax = plt.subplots(figsize=(max(8.27, .35 * len(mean_cols)), max(4.2, .55 * len(profiles))), constrained_layout=True)
                cmap = "Greys" if monochrome else "RdBu_r"
                image = ax.imshow(z, aspect="auto", cmap=cmap, vmin=-2, vmax=2)
                ax.set_xticks(np.arange(len(mean_cols)), [c.removeprefix("mean_") for c in mean_cols], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(profiles)), profiles.cluster_label.astype(str))
                ax.set_title("Standardised cluster profiles", loc="left", fontsize=15, fontweight="bold")
                fig.colorbar(image, ax=ax, label="Profile z-score", shrink=.82)
                _save_figure_set(zf, fig, f"{style}_cluster_profiles")
                plt.close(fig)
        zf.writestr("cluster_assignments.csv", assignments.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("cluster_profiles.csv", profiles.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("cluster_embedding.csv", embedding.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("cluster_diagnostics.csv", diagnostics.to_csv(index=False).encode("utf-8-sig"))
        _write_output_guides(
            zf,
            {
                "Cluster assignments": assignments,
                "Cluster profiles": profiles,
                "Cluster embedding": embedding,
                "Cluster diagnostics": diagnostics,
            },
            ["Cluster projection", "Standardised cluster profiles"],
        )
        zf.writestr("README.txt", "Cluster projection and standardised profiles. Colour and black-and-white; PNG is 600 dpi and SVG/PDF are vector formats.\n")
    return archive.getvalue()


def predictive_publication_bundle(performance: pd.DataFrame, importance: pd.DataFrame, predictions: pd.DataFrame) -> bytes:
    """Publication package for cross-validated model comparison."""
    archive = io.BytesIO()
    perf = performance.sort_values("cross_validated_rmse", ascending=True)
    imp = importance.head(30).sort_values("permutation_importance_mean")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            main = GREYS[1] if monochrome else COLOUR[0]
            accent = GREYS[0] if monochrome else COLOUR[1]
            fig, ax = plt.subplots(figsize=(8.27, 5.7), constrained_layout=True)
            ax.barh(perf.model, perf.cross_validated_rmse, color=main, edgecolor="#222222", hatch="//" if monochrome else "")
            ax.set_xlabel("Cross-validated RMSE (lower is better)"); ax.set_title("Predictive model comparison", loc="left", fontsize=15, fontweight="bold")
            ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_model_comparison")
            plt.close(fig)
            if not imp.empty:
                fig, ax = plt.subplots(figsize=(8.27, max(4.8, .24 * len(imp))), constrained_layout=True)
                ax.barh(imp.variable, imp.permutation_importance_mean, xerr=imp.permutation_importance_sd, color=accent, edgecolor="#222222", hatch="//" if monochrome else "")
                ax.axvline(0, color="#333333", linewidth=.8); ax.set_xlabel("Permutation importance")
                ax.set_title("Predictor importance in the best model", loc="left", fontsize=15, fontweight="bold")
                ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
                _save_figure_set(zf, fig, f"{style}_predictor_importance")
                plt.close(fig)
        zf.writestr("model_performance.csv", performance.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("permutation_importance.csv", importance.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("out_of_fold_predictions.csv", predictions.to_csv(index=False).encode("utf-8-sig"))
        _write_output_guides(
            zf,
            {
                "Cross-validated model performance": performance,
                "Permutation importance": importance,
                "Out-of-fold predictions": predictions,
            },
            ["Predictive model comparison", "Predictor importance"],
        )
        zf.writestr("README.txt", "Cross-validated model performance and permutation importance. These are predictive, not causal, results.\n")
    return archive.getvalue()


def panel_publication_bundle(coefficients: pd.DataFrame, fit: pd.DataFrame, hausman: pd.DataFrame) -> bytes:
    """Publication package for pooled/fixed/random panel coefficients."""
    archive = io.BytesIO()
    data = coefficients[coefficients.term != "const"].copy()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for style, monochrome in (("colour", False), ("black_white", True)):
            palette = GREYS if monochrome else COLOUR
            terms = list(dict.fromkeys(data.term.astype(str)))[:35]
            models = list(dict.fromkeys(data.model.astype(str)))
            fig, ax = plt.subplots(figsize=(8.27, max(5.0, .33 * len(terms))), constrained_layout=True)
            ybase = np.arange(len(terms)); offsets = np.linspace(-.22, .22, max(len(models), 1))
            for i, model in enumerate(models):
                part = data[data.model == model].set_index("term").reindex(terms)
                keep = part.coefficient.notna()
                ypos = ybase[keep] + offsets[i]
                estimates = part.loc[keep, "coefficient"].to_numpy(float)
                lower_error = (part.loc[keep, "coefficient"] - part.loc[keep, "ci_95_low"]).to_numpy(float)
                upper_error = (part.loc[keep, "ci_95_high"] - part.loc[keep, "coefficient"]).to_numpy(float)
                ax.errorbar(estimates, ypos,
                            xerr=np.vstack([lower_error, upper_error]),
                            fmt=MARKERS[i % len(MARKERS)], color=palette[i % len(palette)], capsize=2, label=model)
            ax.axvline(0, color="#333333", linestyle="--", linewidth=.9)
            ax.set_yticks(ybase, terms); ax.set_xlabel("Coefficient and 95% confidence interval")
            ax.set_title("Panel-model coefficient comparison", loc="left", fontsize=15, fontweight="bold")
            ax.legend(frameon=False); ax.grid(axis="x", color="#D9D9D9", linewidth=.55); ax.spines[["top", "right", "left"]].set_visible(False)
            _save_figure_set(zf, fig, f"{style}_panel_coefficients")
            plt.close(fig)
        zf.writestr("panel_coefficients.csv", coefficients.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("panel_fit.csv", fit.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("hausman_test.csv", hausman.to_csv(index=False).encode("utf-8-sig"))
        _write_output_guides(
            zf,
            {"Panel coefficients": coefficients, "Panel fit": fit, "Hausman test": hausman},
            ["Panel-model coefficient comparison"],
        )
        zf.writestr("README.txt", "Panel-model coefficient comparison in colour and black-and-white, with 600-dpi PNG and vector formats.\n")
    return archive.getvalue()
