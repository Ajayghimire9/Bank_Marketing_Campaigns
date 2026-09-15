from __future__ import annotations

import numpy as np


def permutation_importance_report(model, x, y, n_repeats: int = 3) -> list[tuple[str, float]]:
    """Return feature importance without coupling serving to an explanation library."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=42, scoring="roc_auc")
    names = list(x.columns)
    ranking = sorted(zip(names, result.importances_mean), key=lambda item: item[1], reverse=True)
    return [(str(name), float(value)) for name, value in ranking]
