# src/data_prep.py
# Data preprocessing module for Telco Customer Churn dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# Support both the canonical name and the original Kaggle filename
_KAGGLE_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_CSV = (
    os.path.join(DATA_DIR, "telco_churn.csv")
    if os.path.exists(os.path.join(DATA_DIR, "telco_churn.csv"))
    else os.path.join(DATA_DIR, _KAGGLE_NAME)
)
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_churn.csv")


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV from disk."""
    df = pd.read_csv(path)
    print(f"[data_prep] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric; fill NaN with median."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    print(f"[data_prep] TotalCharges: filled {missing} NaN(s) with median={median_val:.2f}")
    return df


def drop_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that do not contribute to prediction (e.g. customerID)."""
    cols_to_drop = [c for c in ["customerID"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[data_prep] Dropped columns: {cols_to_drop}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn Yes→1, No→0."""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print(f"[data_prep] Churn distribution: {df['Churn'].value_counts().to_dict()}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """LabelEncode all remaining object-type columns."""
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=["str", "object"]).columns.tolist()
    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[data_prep] Label-encoded columns: {object_cols}")
    return df


def preprocess(raw_path: str = RAW_CSV, clean_path: str = CLEAN_CSV) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_raw_data(raw_path)
    df = clean_total_charges(df)
    df = drop_non_predictive(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    df.to_csv(clean_path, index=False)
    print(f"[data_prep] Saved cleaned dataset to {clean_path}")
    print(f"[data_prep] Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    preprocess()
