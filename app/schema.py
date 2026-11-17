"""Strict contract for already prepared model inputs; no raw-data imputation."""
import numpy as np
import pandas as pd


def prepare_features(data, names, dtypes):
    if data.empty:
        raise ValueError("No customer rows supplied.")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not accepted.")
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} model features: {', '.join(missing[:8])}")
    result = data.loc[:, names].copy()
    for name in names:
        values = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or non-finite values in {name}.")
        dtype = str(dtypes[name])
        if dtype.startswith(("int", "uint")) and (values != np.floor(values)).any():
            raise ValueError(f"Non-integer values in {name}.")
        if dtype in ("bool", "boolean") and not values.isin([0, 1]).all():
            raise ValueError(f"Expected binary values in {name}.")
        result[name] = values.astype(dtype)
        if not np.array_equal(result[name].to_numpy(dtype=float), values.to_numpy(dtype=float)):
            raise ValueError(f"Lossy conversion in {name}.")
    return result
