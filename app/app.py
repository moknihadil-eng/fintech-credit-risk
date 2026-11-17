import streamlit as st

st.set_page_config(page_title="Credit Risk | Analytics", page_icon=":material/analytics:", layout="wide")
st.markdown("""<style>
.block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px}
[data-testid="stMetricLabel"] {color:#64748b}
h1,h2,h3 {letter-spacing:-.03em}
section[data-testid="stSidebar"] {border-right:1px solid #e2e8f0}
</style>""", unsafe_allow_html=True)
st.sidebar.markdown("## CREDIT RISK")
st.sidebar.caption("Analyse du risque de crédit | Home Credit")
pages = [
    st.Page("pages/dashboard.py", title="Vue d'ensemble", default=True),
    st.Page("pages/client_prediction.py", title="Analyse client"),
    st.Page("pages/model_performance.py", title="Performance du modèle"),
]
st.navigation(pages).run()
