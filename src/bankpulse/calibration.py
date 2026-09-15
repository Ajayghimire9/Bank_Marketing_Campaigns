from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.calibration import calibration_curve


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def reliability_curve(y_true, probabilities, bins: int = 10):
    fraction, mean = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return list(zip(mean.tolist(), fraction.tolist()))
