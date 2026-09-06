# v5.8.6 GitHub / Streamlit update

Upload the v5.8.6 hotfix contents into the existing `makryvelios_dashboard_v2-8` folder, commit, and reboot Streamlit.

For Groq GPT-OSS, keep the new visible **reasoning effort** at `low` for structured RQ and synthesis tasks unless you intentionally want more reasoning and have sufficient token headroom. No automatic model/provider fallback is performed.

# v5.8.5 compact-context Agentic hotfix

Upload the hotfix contents into the existing `makryvelios_dashboard_v2-8` folder, commit to `main`, then reboot Streamlit. Acceptance marker: header **v5.8.5**.

In 12D Agentic Research Mode choose the AI provider/model manually as before. For Groq free-tier start with **AI evidence context = Compact — free-tier friendly** and **AI questions per request = 5 or 10**. The UI now shows an estimated input + configured output token total before Generate. No automatic provider/model fallback is performed.
