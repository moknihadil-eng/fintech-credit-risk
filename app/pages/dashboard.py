"""Vue synthétique du portefeuille de validation."""
import pandas as pd
import streamlit as st
from risk_data import BASE, validation

st.caption("CREDIT RISK / VUE D'ENSEMBLE")
st.title("Le risque de crédit, en perspective")
st.write("Une lecture synthétique des dossiers de validation et des signaux du modèle LightGBM.")
with st.spinner("Chargement du portefeuille..."):
    d = validation()
m = d["metadata"]
y, scores = d["y"], d["scores"]
threshold = float(m["final_threshold"])
flagged = scores >= threshold
cols = st.columns(4)
cols[0].metric("Dossiers de validation", f"{len(y):,}".replace(",", " "))
cols[1].metric("Défauts observés", f"{y.mean():.2%}")
cols[2].metric("Dossiers signalés", f"{flagged.mean():.2%}")
cols[3].metric("ROC-AUC", f"{d['auc']:.4f}")
st.caption(f"échantillon de validation | {len(d['names'])} variables | seuil de classification {threshold:.0%}")
st.divider()
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Distribution des scores")
    bins = pd.cut(scores, bins=[i/10 for i in range(11)], include_lowest=True)
    counts = pd.Series(bins).value_counts(sort=False)
    counts.index = [f"{i*10}-{(i+1)*10} %" for i in range(10)]
    st.bar_chart(counts.rename("Dossiers"), color="#0f766e")
    st.caption("Scores de classification du modèle ; leur calibration en probabilités n'est pas établie.")
with right:
    st.subheader("Détection au seuil retenu")
    tp = int(((y == 1) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    st.metric("Défauts détectés", f"{tp:,}", help="Dossiers en défaut correctement signalés.")
    st.metric("Défauts non détectés", f"{fn:,}")
    st.metric("Fausses alertes", f"{fp:,}")
st.divider()
st.subheader("Les principaux facteurs du modèle")
path = BASE / "data/processed/shap_feature_importance.csv"
if path.exists():
    importance = pd.read_csv(path)
    importance = importance[importance.Feature.isin(d["names"])]
    top = importance.nlargest(12, "Mean_Absolute_SHAP").set_index("Feature")
    st.bar_chart(top["Mean_Absolute_SHAP"], horizontal=True, color="#0f766e")
    st.caption("Importance SHAP enregistrée : contribution absolue moyenne, sans interprétation causale.")
else:
    st.info("L'importance SHAP n'est pas encore disponible.")
with st.expander("Périmètre de l'analyse"):
    st.write(f"Entraînement : {m['n_train_rows']:,} dossiers. Validation : {len(y):,} dossiers. Les indicateurs décrivent ce jeu de validation et ne constituent pas une mesure de performance en production.")
