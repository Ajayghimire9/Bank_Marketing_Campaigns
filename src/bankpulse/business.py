from __future__ import annotations

import numpy as np


def campaign_policy(probabilities, contact_cost: float = 1.0, success_value: float = 10.0):
    """Convert propensity into an economically motivated outreach decision."""
    p = np.asarray(probabilities, dtype=float)
    expected_value = p * success_value - contact_cost
    return expected_value, (expected_value > 0).astype(int)


def uplift_report(y_true, probabilities, budget_fraction: float = 0.20) -> dict[str, float]:
    """Measure how much positive response is captured by the highest-ranked cohort."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    k = max(1, int(len(y) * budget_fraction))
    order = np.argsort(-p)
    top_rate = float(y[order[:k]].mean())
    overall = float(y.mean())
    return {"top_cohort_rate": top_rate, "overall_rate": overall, "lift": top_rate / overall if overall else 0.0}
