import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from risk_data import BASE, validation

st.caption("CREDIT RISK / PERFORMANCE")
st.title("Mesurer la qualité de détection")
st.write("Explorez le compromis entre détection des défauts et fausses alertes sur le jeu de validation.")
with st.expander("Comparaison historique des modeles"):
    comparison = BASE / "reports/model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), hide_index=True, use_container_width=True)
        st.caption("Resultats sauvegardes, sans nouvel entrainement. Les variantes, scores CV et scores de validation sont distincts. Le resultat optimise de la regression logistique n'est pas enregistre ; un protocole identique entre tous les essais n'est pas etabli.")
d = validation()
y, scores = d["y"], d["scores"]
operational = float(d["metadata"]["final_threshold"])
cols = st.columns(4)
cols[0].metric("ROC-AUC", f"{d['auc']:.4f}")
cols[1].metric("KS", f"{d['ks']:.4f}")
cols[2].metric("Average precision", f"{average_precision_score(y, scores):.4f}")
cols[3].metric("Défauts observés", f"{y.mean():.2%}")
simulation, curves = st.tabs(["Simulation du seuil", "Courbe ROC"])
with simulation:
    threshold = st.slider("Seuil de classification", 0.0, 1.0, operational, 0.01)
    st.caption(f"Seuil enregistré : {operational:.0%}. La simulation ne modifie pas le modèle.")
    pred = scores >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cols = st.columns(4)
    cols[0].metric("Précision | défaut", f"{precision_score(y,pred,zero_division=0):.2%}")
    cols[1].metric("Rappel | défaut", f"{recall_score(y,pred,zero_division=0):.2%}")
    cols[2].metric("F1 | défaut", f"{f1_score(y,pred,zero_division=0):.4f}")
    cols[3].metric("Dossiers signalés", f"{int(pred.sum()):,}")
    st.subheader("Résultats de classification")
    st.dataframe(pd.DataFrame(cm, index=["Réel : sans défaut", "Réel : défaut"], columns=["Prédit : sans défaut", "Prédit : défaut"]), use_container_width=True)
    st.info(f"{cm[1,1]:,} défauts détectés | {cm[1,0]:,} défauts non détectés | {cm[0,1]:,} fausses alertes.")
    st.caption("Le choix du seuil dépend du coût des défauts manqués et des fausses alertes. Aucun optimum économique n'est établi ici.")
    st.download_button("Exporter les résultats", pd.DataFrame({"observed_default":y,"model_score":scores,"predicted_default":pred.astype(int)}).to_csv(index=False), "validation_predictions.csv", "text/csv")
with curves:
    indices = np.unique(np.linspace(0, len(d['fpr'])-1, min(1000,len(d['fpr']))).astype(int))
    st.line_chart(pd.DataFrame({"Faux positifs":d['fpr'][indices],"Défauts détectés":d['tpr'][indices]}), x="Faux positifs", y="Défauts détectés", color="#0f766e")
    st.caption("Axe horizontal : taux de faux positifs. Axe vertical : taux de vrais positifs.")
