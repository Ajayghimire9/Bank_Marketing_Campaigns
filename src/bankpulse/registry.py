from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(model_path: str | Path, metrics: dict, output: str | Path) -> None:
    payload = {
        "model_version": "bankpulse-1.0.0",
        "artifact": str(model_path),
        "sha256": sha256(model_path),
        "metrics": metrics,
    }
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def promotion_allowed(metrics: dict, minimum_roc_auc: float = 0.70) -> bool:
    return float(metrics.get("roc_auc", 0.0)) >= minimum_roc_auc
