from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_pipeline


def candidates(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic-regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        ),
        "hist-gradient-boosting": build_pipeline(
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=random_state)
        ),
    }


def save(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
