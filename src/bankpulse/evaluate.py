from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def select_threshold(y_true, probabilities) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.10, 0.90, 81):
        score = f1_score(y_true, np.asarray(probabilities) >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold
