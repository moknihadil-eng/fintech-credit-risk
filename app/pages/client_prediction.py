"""Individual scoring using complete, already prepared feature vectors."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from risk_data import BASE, validation
from schema import prepare_features

st.caption("CREDIT RISK / ANALYSE CLIENT")
st.title("Analyser un dossier client")
st.write("Selectionnez un dossier de validation ou importez des donnees deja preparees avec le schema du modele. Un CSV application brut peut aussi etre transforme avec le pipeline historique verifie.")
d = validation()
model, metadata, feature_names = d["model"], d["metadata"], d["names"]
threshold = float(metadata["final_threshold"])
feature_dtypes = joblib.load(BASE / "data/models/lightgbm_feature_dtypes.pkl")

@st.cache_data
def load_prepared_validation():
    _, x, _, _ = joblib.load(BASE / "data/processed/xgb_split.pkl")
    return prepare_features(x, feature_names, feature_dtypes)

X_valid = load_prepared_validation()
mode = st.radio("Source du dossier", ["Dossier de validation", "CSV prepare", "CSV application brut"], horizontal=True)
if mode == "Dossier de validation":
    row = st.number_input("Position du dossier (index a partir de zero)", min_value=0, max_value=len(X_valid)-1, value=0, step=1)
    client = X_valid.iloc[[int(row)]].reset_index(drop=True)
    st.caption("Dossier reel du jeu de validation : aucune variable remplacee par une mediane.")
else:
    if mode == "CSV application brut":
        example = pd.read_csv(BASE / "data/raw/application_train.csv", nrows=1).drop(columns="TARGET", errors="ignore")
        st.download_button("Telecharger un exemple brut", example.to_csv(index=False), "raw_client_example.csv", "text/csv")
        st.info("Les historiques proviennent des tables locales. Pour un nouveau client, ces tables doivent etre actualisees avant le calcul ; le pipeline ne recupere pas son historique bancaire.")
    else:
        st.download_button("Telecharger un exemple au bon format", X_valid.head(1).to_csv(index=False), "prepared_client_example.csv", "text/csv")
    uploaded = st.file_uploader("CSV du dossier client", type="csv")
    if uploaded is None:
        st.stop()
    try:
        data = pd.read_csv(uploaded)
        if mode == "CSV application brut":
            from raw_pipeline import prepare_raw
            data = prepare_raw(data)
            st.caption("Historique issu des six tables agregees locales, joint par SK_ID_CURR. Aucun historique externe n'est recupere automatiquement.")
        prepared = prepare_features(data, feature_names, feature_dtypes)
        row = st.number_input("Ligne a analyser", min_value=0, max_value=len(prepared)-1, value=0, step=1)
        client = prepared.iloc[[int(row)]].reset_index(drop=True)
    except (KeyError, FileNotFoundError, ValueError, TypeError, OverflowError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Fichier refuse : {error}")
        st.stop()
st.caption(f"{len(feature_names)} variables controlees : presence, ordre, types et valeurs finies.")
with st.expander("Variables du dossier"):
    st.dataframe(client, use_container_width=True, hide_index=True)
if not st.button("Calculer le risque", type="primary"):
    st.stop()
probability = model.predict_proba(client)[0, 1]

prediction = int(
    probability >= threshold
)


# ============================================================
# RÉSULTAT
# ============================================================

st.divider()

st.subheader("📊 Résultat de la prédiction")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Score de défaut",
        f"{probability:.2%}"
    )


with col2:

    st.metric(
        "Seuil de décision",
        f"{threshold:.2f}"
    )


with col3:

    if prediction == 1:

        st.metric(
            "Décision",
            "🔴 Risque élevé"
        )

    else:

        st.metric(
            "Décision",
            "🟢 Risque faible"
        )


# ============================================================
# MESSAGE DE DÉCISION
# ============================================================

if prediction == 1:

    st.error(
        f"""
        **RISQUE ÉLEVÉ**

        Le client saisi présente une score
        de défaut de **{probability:.2%}**, supérieure au
        seuil de décision de **{threshold:.2f}**.
        """
    )

else:

    st.success(
        f"""
        **RISQUE FAIBLE**

        Le client saisi présente une score
        de défaut de **{probability:.2%}**, inférieure au
        seuil de décision de **{threshold:.2f}**.
        """
    )


# ============================================================
# SHAP — CLIENT SÉLECTIONNÉ
# ============================================================

st.divider()

st.subheader("🧠 Pourquoi cette décision ?")

st.write(
    """
    L'analyse SHAP permet d'identifier les variables qui
    augmentent ou diminuent le risque prédit pour ce client.
    """
)


# ============================================================
# CRÉATION DE L'EXPLAINER
# ============================================================

@st.cache_resource
def create_explainer():

    return shap.TreeExplainer(model)


explainer = create_explainer()


# ============================================================
# CALCUL SHAP
# ============================================================

shap_values = explainer.shap_values(client)


# ============================================================
# GESTION DES VERSIONS SHAP
# ============================================================

if isinstance(shap_values, list):

    shap_values_client = shap_values[1]

else:

    shap_values_client = shap_values


# ============================================================
# VALEUR DE BASE
# ============================================================

if isinstance(explainer.expected_value, list):

    base_value = explainer.expected_value[1]

else:

    base_value = explainer.expected_value


# ============================================================
# VALEURS SHAP 1D
# ============================================================

shap_values_1d = np.asarray(
    shap_values_client
).reshape(-1)


client_values = client.iloc[0].values


# ============================================================
# TABLEAU SHAP
# ============================================================

client_shap = pd.DataFrame({

    "Feature": X_valid.columns,

    "Value": client_values,

    "SHAP_Value": shap_values_1d

})


client_shap["Abs_SHAP"] = (
    client_shap["SHAP_Value"].abs()
)


client_shap = (
    client_shap
    .sort_values(
        "Abs_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# TOP VARIABLES EXPLICATIVES
# ============================================================

st.subheader("🔍 Principales variables explicatives")


top_features = client_shap.head(10).copy()


st.dataframe(
    top_features[
        [
            "Feature",
            "Value",
            "SHAP_Value"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INTERPRÉTATION AUTOMATIQUE
# ============================================================

top_positive = (
    client_shap[
        client_shap["SHAP_Value"] > 0
    ]
    .sort_values(
        "SHAP_Value",
        ascending=False
    )
    .head(1)
)


top_negative = (
    client_shap[
        client_shap["SHAP_Value"] < 0
    ]
    .sort_values(
        "SHAP_Value",
        ascending=True
    )
    .head(1)
)


st.subheader("💡 Interprétation métier")


if not top_positive.empty:

    feature_positive = (
        top_positive.iloc[0]["Feature"]
    )

    shap_positive = (
        top_positive.iloc[0]["SHAP_Value"]
    )

    st.write(
        f"• **{feature_positive}** contribue à "
        f"**augmenter le risque prédit** "
        f"(SHAP = {shap_positive:.4f})."
    )


if not top_negative.empty:

    feature_negative = (
        top_negative.iloc[0]["Feature"]
    )

    shap_negative = (
        top_negative.iloc[0]["SHAP_Value"]
    )

    st.write(
        f"• **{feature_negative}** contribue à "
        f"**réduire le risque prédit** "
        f"(SHAP = {shap_negative:.4f})."
    )


st.caption(
    """
    Une valeur SHAP positive contribue à augmenter
    le risque de défaut prédit. Une valeur SHAP négative
    contribue à le réduire.
    """
)


# ============================================================
# WATERFALL SHAP
# ============================================================

st.subheader("📈 Waterfall SHAP")

st.write(
    """
    Le graphique montre comment les principales variables
    font évoluer la prédiction du modèle depuis sa valeur
    de base jusqu'à la prédiction finale du client.
    """
)


fig = plt.figure(
    figsize=(10, 7)
)


shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_1d,
        base_values=base_value,
        data=client.iloc[0].values,
        feature_names=X_valid.columns.tolist()
    ),
    max_display=15,
    show=False
)


st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# INFORMATIONS TECHNIQUES
# ============================================================

st.divider()

with st.expander("ℹ️ Informations techniques"):

    st.write("**Source des données :** dossier prepare complet")

    st.write(
        f"**Nombre de features :** "
        f"{X_valid.shape[1]}"
    )

    st.write(
        f"**Score :** "
        f"{probability:.6f}"
    )

    st.write(
        f"**Seuil :** "
        f"{threshold:.2f}"
    )

    st.write(
        f"**Classe prédite :** "
        f"{prediction}"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    f"LightGBM + scale_pos_weight | "
    f"Client saisi | "
    f"Seuil final : {threshold:.2f}"
)
