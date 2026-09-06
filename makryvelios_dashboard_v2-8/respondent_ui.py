"""Cloud-only wrapper for respondent analytics + confirmatory/rare methods.

This file intentionally requires no local patch script. Upload it together with
respondent_ui_core.py, confirmatory_ui.py, confirmatory_analytics.py and
analysis_recipes.py to the existing Streamlit app directory in GitHub.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from respondent_ui_core import render_respondent_module as _render_respondent_core
from confirmatory_ui import render_confirmatory_lab


def render_respondent_module(df: pd.DataFrame, source_label: str = "Active dataset") -> None:
    st.subheader("Expert respondent analytics · extended research laboratory")
    st.caption(
        "Cloud extension: the original respondent/ITA workflow is preserved, and the generalised confirmatory & rare-methods engine is available beside it. Computation runs on the Streamlit server; users need only a web browser."
    )
    respondent_tab, confirmatory_tab = st.tabs([
        "Original respondent / ITA analytics",
        "Generalised confirmatory & rare methods",
    ])
    with respondent_tab:
        _render_respondent_core(df, source_label)
    with confirmatory_tab:
        render_confirmatory_lab(df)
