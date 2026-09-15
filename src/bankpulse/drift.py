from __future__ import annotations

import numpy as np


def psi(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_status(score: float) -> str:
    if score >= 0.25:
        return "critical"
    if score >= 0.10:
        return "warning"
    return "stable"
