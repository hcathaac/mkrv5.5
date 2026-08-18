"""Portable HTML reporting and ZIP helpers."""
from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Mapping, Sequence

import pandas as pd

from output_guidance import explain_output


STYLE = """
:root{--ink:#17212b;--blue:#155b8a;--gold:#d89b2b;--paper:#fff;--muted:#65727e;--line:#dbe2e8}
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.55 Arial,sans-serif}
main{max-width:1180px;margin:28px auto;background:var(--paper);padding:44px 52px;box-shadow:0 8px 30px #1b2b3815}
h1{font-size:30px;line-height:1.15;margin:0 0 8px;color:var(--blue)}h2{margin-top:38px;border-bottom:2px solid var(--gold);padding-bottom:7px}
.meta{color:var(--muted);font-size:13px}.callout{border-left:4px solid var(--gold);background:#fbf7ee;padding:12px 16px;margin:14px 0}
.table-wrap{overflow:auto;border:1px solid var(--line);margin:14px 0 25px}table{border-collapse:collapse;width:100%;font-size:12px}
th{background:#eaf1f6;color:#173b55;position:sticky;top:0}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
tr:nth-child(even) td{background:#fafcfd}.chart{margin:22px 0;border:1px solid var(--line);padding:8px}
footer{margin-top:42px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media(max-width:760px){main{margin:0;padding:24px 18px;box-shadow:none}h1{font-size:24px}}
@media print{body{background:white}main{box-shadow:none;margin:0;max-width:none}.table-wrap{overflow:visible}.chart{break-inside:avoid}}
"""


def build_html_report(
    title: str,
    source_note: str,
    tables: Mapping[str, pd.DataFrame],
    chart_html: Sequence[str] = (),
    comments: Sequence[str] = (),
) -> bytes:
    sections = []
    if comments:
        sections.append("<h2>Interpretive comments</h2>" + "".join(f'<div class="callout">{html.escape(str(c))}</div>' for c in comments))
    for name, table in tables.items():
        bounded = table.head(10_000)
        guide = explain_output(name, table, "table")
        note = "<p class='meta'>Displayed/exported in this report: first 10,000 rows.</p>" if len(table) > len(bounded) else ""
        explanation = (
            f"<div class='callout'><strong>Plain-language meaning.</strong> {html.escape(guide['plain_language_summary'])}<br>"
            f"<strong>How to read it.</strong> {html.escape(guide['how_to_read_it'])}<br>"
            f"<strong>What it does not mean.</strong> {html.escape(guide['what_it_does_not_mean'])}<br>"
            f"<strong>Next step.</strong> {html.escape(guide['recommended_next_step'])}</div>"
        )
        sections.append(f"<h2>{html.escape(name)}</h2>{explanation}{note}<div class='table-wrap'>{bounded.to_html(index=False, border=0, escape=True)}</div>")
    if chart_html:
        chart_guide = explain_output("Report figures", None, "chart")
        sections.append(
            "<h2>Figures</h2>"
            f"<div class='callout'><strong>How to read the figures.</strong> {html.escape(chart_guide['how_to_read_it'])}<br>"
            f"<strong>What they do not mean.</strong> {html.escape(chart_guide['what_it_does_not_mean'])}</div>"
            + "".join(f'<div class="chart">{c}</div>' for c in chart_html)
        )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main><header><h1>{html.escape(title)}</h1><div class="meta">Generated {generated}</div><p>{html.escape(source_note)}</p></header>{''.join(sections)}<footer>Reproducible analytical output. Statistical association does not by itself establish causality. Verify coding, estimand, sampling and identification assumptions before publication.</footer></main></body></html>"""
    return document.encode("utf-8")
