"""Données et prédictions communes aux vues de validation."""
from pathlib import Path
import re
import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parents[1]

@st.cache_resource
def validation():
    model = joblib.load(BASE / "data/models/lightgbm_scale_pos_weight.pkl")
    metadata = joblib.load(BASE / "data/models/lightgbm_metadata.pkl")
    names = joblib.load(BASE / "data/models/lightgbm_feature_names.pkl")
    _, x, _, y = joblib.load(BASE / "data/processed/xgb_split.pkl")
    from schema import prepare_features
    dtypes = joblib.load(BASE / "data/models/lightgbm_feature_dtypes.pkl")
    x = prepare_features(x, names, dtypes)
    if model.n_features_in_ != len(names):
        raise ValueError("Model and saved feature schema disagree.")
    scores = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, scores)
    return dict(model=model, metadata=metadata, names=names, y=np.asarray(y),
                scores=scores, fpr=fpr, tpr=tpr, thresholds=thresholds,
                auc=roc_auc_score(y, scores), ks=float(np.max(tpr-fpr)))
