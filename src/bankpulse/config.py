from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    test_size: float = 0.2
    decision_threshold: float = 0.50
    min_roc_auc: float = 0.70
    model_version: str = "bankpulse-1.0.0"
