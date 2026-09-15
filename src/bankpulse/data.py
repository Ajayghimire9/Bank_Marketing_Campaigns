from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "y"
FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor3m", "nr_employed",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";" if str(path).endswith(".csv") else ",")
    frame.columns = [c.strip().lower().replace(".", "_") for c in frame.columns]
    if TARGET not in frame:
        raise ValueError("Dataset must contain target column 'y'")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return frame[FEATURES + [TARGET]].copy()


def validate(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame[TARGET].dropna().isin(["yes", "no", 0, 1]).all() is False:
        raise ValueError("Target must contain yes/no or 0/1 values")
