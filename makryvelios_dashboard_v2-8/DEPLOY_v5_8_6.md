# Makryvelios v5.8.6 deployment note

Apply on top of v5.8.5. Upload the hotfix contents directly into `makryvelios_dashboard_v2-8`, preserving the existing folder structure, commit, then reboot the Streamlit app.

For Groq `openai/gpt-oss-120b`, leave **Groq GPT-OSS reasoning effort = low** for RQ generation and paper synthesis. The setting is visible and manual; the application never switches provider or model automatically.
