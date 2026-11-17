"""Recover historical parameters and verify parity before publishing artifact."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from preprocessing import TABLES, base_frame, recover_specs, transform_raw

BASE = Path(__file__).resolve().parents[1]


def main():
    train, valid, _, _ = joblib.load(BASE / "data/processed/xgb_split.pkl")
    raw = pd.read_csv(BASE / "data/raw/application_train.csv")
    tables = {name: pd.read_csv(BASE / f"data/processed/{name}.csv") for name in TABLES}
    names = joblib.load(BASE / "data/models/lightgbm_feature_names.pkl")
    dtypes = joblib.load(BASE / "data/models/lightgbm_feature_dtypes.pkl")
    application = raw.loc[train.index].reset_index(drop=True)
    specs = recover_specs(base_frame(application, tables), train[names])
    artifact = dict(names=names, dtypes=dtypes, specs=specs,
                    provenance="Recovered from saved training rows; no refit of historical model.")
    report = {}
    for label, expected in [("train", train), ("validation", valid)]:
        rebuilt = transform_raw(raw.loc[expected.index].reset_index(drop=True), tables, artifact)
        np.testing.assert_allclose(rebuilt.to_numpy(float), expected[names].to_numpy(float), rtol=1e-12, atol=1e-12)
        report[label] = {"rows": len(expected), "features": len(names), "parity": True}
    (BASE / "data/models/preprocessing.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (BASE / "reports/preprocessing_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
