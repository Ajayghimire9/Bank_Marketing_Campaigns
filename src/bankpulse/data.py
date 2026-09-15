from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "y"
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]


def load_dataset(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [c.strip().lower().replace(".", "_") for c in frame.columns]
    if require_target and TARGET not in frame:
        raise ValueError("Dataset must contain target column 'y'")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return frame[FEATURES + ([TARGET] if TARGET in frame else [])].copy()


def validate(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Dataset is empty")
    encode_target(frame[TARGET])
    target = frame[TARGET].astype(str).str.strip().str.lower()
    if not target.isin(["yes", "no", "0", "1"]).all():
        raise ValueError("Target must contain yes/no or 0/1 values")
    if frame[FEATURES].isnull().all(axis=1).any():
        raise ValueError("Rows with every feature missing are not allowed")


def encode_target(target: pd.Series) -> pd.Series:
    """Use one label mapping for numeric and textual datasets."""
    result = target.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if result.isna().any():
        raise ValueError("Target contains missing or unsupported labels")
    return result.astype(int)
