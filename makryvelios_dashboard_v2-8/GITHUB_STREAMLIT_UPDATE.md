# Update the GitHub and Streamlit app

The folder name remains `makryvelios_dashboard_v2` so the existing Streamlit main-file path does not need to change.

## Browser-only update

1. Download and extract `Makryvelios_Research_Analytics_v5_6_1.zip`.
2. Open the GitHub repository `hchataac/testrdmakrv2`.
3. Remove the old `makryvelios_dashboard_v2` folder, or upload the new folder contents with the same filenames and confirm replacement.
4. Confirm that `makryvelios_dashboard_v2/app.py`, `analytics_core.py`, `advanced_analytics.py`, `mcda.py`, `research_chair.py`, `prompt_library.py`, `output_guidance.py`, `visuals.py`, `requirements.txt`, `source_evidence_catalogue.csv`, the `documentation` folder and the `data` folder are present.
5. Commit the changes to the `main` branch.
6. Open the Streamlit Community Cloud workspace and select the existing app.
7. Keep the main file path as `makryvelios_dashboard_v2/app.py`.
8. Streamlit normally rebuilds after the GitHub commit. If it does not, open **Manage app**, choose **Reboot app**, and inspect the logs.

## What should be visible after rebuilding

- The dark blue/cyan high-technology header marked **POSTDOCTORAL ANALYTICAL ENGINE v5.6.1**.
- The data intake console option **Combine columns side-by-side (by row order)** and a multi-dataset selector with all sources selected by default.
- The sidebar module **6. OLS & econometric laboratory**.
- The sidebar module **6A. Monte Carlo & uncertainty**.
- The additional **8A Panel model laboratory**, **10A Advanced clustering & segmentation**, and **10B Predictive model laboratory** modules.
- The new **12A Dedicated MCDA engine** with MAVT, TOPSIS, PROMETHEE II, AHP/Entropy/CRITIC weighting and Monte Carlo rank acceptability.
- The separate **12A.1 ITA / public-funding decision support** module with ITA-PB, Hybrid ITA-RW, exact portfolio optimisation, modern round/scenario visualisations, Greek regional maps and GAMS-ready exports.
- **12A.2 Expert respondent analytics** with complete distributions, concordance, PCA, preference segments, subgroup tests and a direct empirical-weight bridge to Hybrid ITA-RW.
- **12B Research Command Chair** with XLSX/PDF scoping, algorithms, equations, natural-language interpretation, a fifth copy-ready Prompt Library tab and expanded paper-report bundles.
- The downloadable v5.2.1 documentation library under **13 Methods & reproducibility**.
- The complete retained v5.5.3 module set plus the new ITA decision-support module.

## Confidentiality

The package includes `data/rd_projects_reference.xlsx`. Do not keep that workbook in a public repository if it is confidential. The application can instead receive the workbook through its upload control at run time.
