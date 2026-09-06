# Makryvelios v5.8.5 deployment

Additive hotfix over v5.8.4. Upload the hotfix contents into the existing `makryvelios_dashboard_v2-8` directory and reboot Streamlit.

Key behaviour: Agentic AI calls now use a user-visible context profile and bounded request size. No provider/model fallback is automatic. For Groq free-tier use, start with **Compact — free-tier friendly**, 5–10 questions per AI request, and review the visible token estimate before Generate.
