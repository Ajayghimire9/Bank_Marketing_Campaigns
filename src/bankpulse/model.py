from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_pipeline


def candidates(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic-regression": build_pipeline(
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state)
        ),
        "random-forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "hist-gradient-boosting": build_pipeline(
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=random_state,
            )
        ),
    }


def save(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
