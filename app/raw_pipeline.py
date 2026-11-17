"""Adapter for the verified notebook preprocessing artifact."""
import json
import sys
import pandas as pd
import streamlit as st
from risk_data import BASE

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from src.preprocessing import TABLES, transform_raw


@st.cache_resource
def load_raw_pipeline():
    artifact = json.loads((BASE / "data/models/preprocessing.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_csv(BASE / f"data/processed/{name}.csv") for name in TABLES}
    return artifact, tables


def prepare_raw(application):
    artifact, tables = load_raw_pipeline()
    return transform_raw(application, tables, artifact)
