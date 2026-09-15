from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import TARGET, load_dataset
from .model import load


def score(data_path: str, model_path: str, output: str) -> None:
    frame = load_dataset(data_path)
    features = frame.drop(columns=[TARGET])
    model = load(model_path)
    frame["propensity"] = model.predict_proba(features)[:, 1]
    frame["prediction"] = (frame["propensity"] >= 0.5).astype(int)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output", default="artifacts/scored.parquet")
    args = parser.parse_args()
    score(args.data, args.model, args.output)
