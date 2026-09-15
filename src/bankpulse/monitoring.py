from __future__ import annotations

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("bankpulse_prediction_requests_total", "Prediction requests")
ERRORS = Counter("bankpulse_prediction_errors_total", "Prediction errors")
LATENCY = Histogram("bankpulse_prediction_latency_seconds", "Prediction latency")
POSITIVE_RATE = Gauge("bankpulse_prediction_positive_rate", "Current positive prediction rate")


def prediction_drift(reference, current) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    return float(abs(current.mean() - reference.mean()))
