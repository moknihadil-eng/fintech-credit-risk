"""Réentraîne et publie les artefacts LightGBM sans la variable CODE_GENDER."""

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_PATH = PROCESSED_DIR / "xgb_split.pkl"
EXCLUDED_FEATURES = ["CODE_GENDER"]
FINAL_THRESHOLD = 0.68


def main():
    X_train, X_valid, y_train, y_valid = joblib.load(SPLIT_PATH)

    missing_features = set(EXCLUDED_FEATURES) - set(X_train.columns)
    if missing_features:
        raise ValueError(f"Variables introuvables : {sorted(missing_features)}")

    X_train = X_train.drop(columns=EXCLUDED_FEATURES)
    X_valid = X_valid.drop(columns=EXCLUDED_FEATURES)
    # LightGBM n'accepte pas certains caractères des noms de colonnes.
    cleaned_columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(column)) for column in X_train.columns]
    X_train.columns = cleaned_columns
    X_valid.columns = cleaned_columns
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=50,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= FINAL_THRESHOLD).astype(int)
    fpr, tpr, thresholds = roc_curve(y_valid, probabilities)
    ks_values = tpr - fpr
    ks_index = int(np.argmax(ks_values))
    roc_auc = roc_auc_score(y_valid, probabilities)
    ks = float(ks_values[ks_index])
    cm = confusion_matrix(y_valid, predictions)

    metadata = {
        "model_name": "LightGBM",
        "model_variant": "scale_pos_weight_without_code_gender",
        "excluded_features": EXCLUDED_FEATURES,
        "n_features": X_train.shape[1],
        "n_train_rows": X_train.shape[0],
        "n_valid_rows": X_valid.shape[0],
        "final_threshold": FINAL_THRESHOLD,
        "roc_auc": round(float(roc_auc), 4),
        "ks": round(ks, 4),
    }

    joblib.dump(model, MODELS_DIR / "lightgbm_scale_pos_weight.pkl", compress=3)
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "lightgbm_feature_names.pkl")
    joblib.dump(X_train.dtypes.astype(str).to_dict(), MODELS_DIR / "lightgbm_feature_dtypes.pkl")
    joblib.dump(metadata, MODELS_DIR / "lightgbm_metadata.pkl")

    evaluation = pd.DataFrame([{
        "ROC_AUC": roc_auc,
        "KS": ks,
        "KS_threshold": thresholds[ks_index],
        "Final_threshold": FINAL_THRESHOLD,
        "Final_accuracy": accuracy_score(y_valid, predictions),
        "Final_precision": precision_score(y_valid, predictions, zero_division=0),
        "Final_recall": recall_score(y_valid, predictions, zero_division=0),
        "Final_F1": f1_score(y_valid, predictions, zero_division=0),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }])
    evaluation.to_csv(PROCESSED_DIR / "model_final_evaluation.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Aléatoire")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC — LightGBM sans CODE_GENDER")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "lightgbm_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds[:-1], ks_values[:-1], label="KS")
    plt.axvline(thresholds[ks_index], linestyle="--", label=f"Seuil KS = {thresholds[ks_index]:.2f}")
    plt.xlabel("Seuil")
    plt.ylabel("KS")
    plt.title("Courbe KS — LightGBM sans CODE_GENDER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "lightgbm_ks_curve.png", dpi=150)
    plt.close()

    X_shap = X_valid.sample(n=min(2000, len(X_valid)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = pd.DataFrame({
        "Feature": X_shap.columns,
        "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values("Mean_Absolute_SHAP", ascending=False)
    shap_importance.to_csv(PROCESSED_DIR / "shap_feature_importance.csv", index=False)
    joblib.dump(shap_values, PROCESSED_DIR / "shap_values_default.pkl")
    X_shap.to_csv(PROCESSED_DIR / "X_shap_2000.csv", index=False)

    print(f"Variables utilisées : {X_train.shape[1]}")
    print(f"CODE_GENDER présente : {'CODE_GENDER' in X_train.columns}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"KS : {ks:.4f}")


if __name__ == "__main__":
    main()
