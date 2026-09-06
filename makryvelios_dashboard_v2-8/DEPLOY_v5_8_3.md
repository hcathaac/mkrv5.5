# Deploy v5.8.3 over v5.8.2

This is an additive hotfix. Keep the existing repository, branch and Streamlit entry point. Upload the hotfix contents into the existing `makryvelios_dashboard_v2-8` directory and confirm replacement of same-name files. Commit to `main` and reboot the Streamlit app.

After deployment, verify:

1. Header shows v5.8.3.
2. Sidebar AI panel shows **TEST AI CONNECTION** when an engine is configured.
3. In 12D Agentic Research Mode, Gemini RQ generation no longer fails solely because a response is not parseable JSON.
4. If an AI batch needs recovery, the UI reports AI-grounded and deterministic-recovery counts instead of aborting the workflow.
5. All v5.8.2 and earlier modules remain visible.
