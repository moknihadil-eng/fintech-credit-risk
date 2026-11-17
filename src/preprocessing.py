"""Frozen reconstruction of the historical notebook transformations.

Parameters are recovered from training rows only. Export is permitted only
after equality with the saved training and validation feature matrices.
Historical aggregate tables must be supplied explicitly.
"""
import numpy as np
import pandas as pd

TABLES = ["bureau_features", "bureau_balance_features", "installments_features",
          "previous_application_features", "pos_cash_features", "credit_card_features"]
CATEGORIES = ["NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
              "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
              "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
              "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "age_tranche"]


def base_frame(application, tables):
    if application.empty or "SK_ID_CURR" not in application:
        raise ValueError("Raw application requires SK_ID_CURR and at least one row.")
    if application.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate customer IDs.")
    data = application.copy()
    for name in TABLES:
        table = tables[name]
        data = data.merge(table, on="SK_ID_CURR", how="left", validate="one_to_one")
    data["age_years"] = (-data.DAYS_BIRTH / 365).astype(int)
    data["age_tranche"] = pd.cut(data.age_years, [20, 30, 40, 50, 60, 70],
                                labels=["20-30", "30-40", "40-50", "50-60", "60-70"],
                                include_lowest=True).astype(str)
    data["OWN_CAR_AGE_MISSING"] = data.OWN_CAR_AGE.isna().astype(int)
    data["DAYS_EMPLOYED_ANOM"] = data.DAYS_EMPLOYED.eq(365243).astype(int)
    data["DAYS_EMPLOYED"] = data.DAYS_EMPLOYED.replace(365243, np.nan)
    data["AMT_INCOME_TOTAL_LOG"] = np.log1p(data.AMT_INCOME_TOTAL)
    return data


def keys(values):
    return values.astype("string").fillna("__MISSING__")


def recover_specs(base, expected):
    specs = {}
    for feature in expected:
        source = next((c for c in CATEGORIES if feature.startswith(c + "_")), feature)
        if source not in base:
            raise ValueError(f"Cannot reconstruct source for {feature}")
        target = expected[feature].reset_index(drop=True)
        values = base[source].reset_index(drop=True)
        if target.nunique() == 1 and feature in ["DAYS_EMPLOYED_ANOM", "OWN_CAR_AGE_MISSING"]:
            specs[feature] = {"source": source, "constant": float(target.iloc[0])}
            continue
        if source in CATEGORIES or not pd.api.types.is_numeric_dtype(values):
            pairs = pd.DataFrame({"key": keys(values), "value": target})
            if pairs.groupby("key").value.nunique().max() > 1:
                raise ValueError(f"Ambiguous encoding for {feature}")
            specs[feature] = {"source": source, "mapping": pairs.groupby("key").value.first().to_dict()}
            continue
        valid = values.notna()
        if not np.allclose(values[valid], target[valid], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Unreconstructed numeric transformation: {feature}")
        spec = {"source": source}
        missing = ~valid
        if missing.any():
            if target[missing].nunique() == 1:
                spec["fill"] = float(target[missing].iloc[0])
            else:
                group = "age_tranche" if feature in ["EXT_SOURCE_1", "EXT_SOURCE_3"] else "FLAG_OWN_CAR"
                pairs = pd.DataFrame({"key": keys(base[group].reset_index(drop=True))[missing], "value": target[missing]})
                if pairs.groupby("key").value.nunique().max() > 1:
                    raise ValueError(f"Ambiguous imputation for {feature}")
                spec.update(group=group, fills=pairs.groupby("key").value.first().to_dict())
        specs[feature] = spec
    return specs


def transform_raw(application, tables, artifact):
    base = base_frame(application, tables)
    result = {}
    for feature, spec in artifact["specs"].items():
        values = base[spec["source"]]
        if "constant" in spec:
            values = pd.Series(spec["constant"], index=base.index)
        elif "mapping" in spec:
            values = keys(values).map(spec["mapping"])
        elif "group" in spec:
            values = values.fillna(keys(base[spec["group"]]).map(spec["fills"]))
        elif "fill" in spec:
            values = values.fillna(spec["fill"])
        if values.isna().any():
            raise ValueError(f"Unknown category or unavailable historical imputation: {feature}")
        result[feature] = values.astype(artifact["dtypes"][feature])
    return pd.DataFrame(result).loc[:, artifact["names"]]
