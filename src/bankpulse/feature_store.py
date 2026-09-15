from __future__ import annotations

from pathlib import Path

import pandas as pd


class OfflineFeatureStore:
    """Small local feature-store abstraction with point-in-time-safe batch reads."""

    def __init__(self, root: str = "feature_store") -> None:
        self.root = Path(root)

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / f"{name}.parquet")
