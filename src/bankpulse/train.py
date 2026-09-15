from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import yaml
from sklearn.model_selection import train_test_split

from .data import TARGET, load_dataset, validate
from .evaluate import evaluate, select_threshold
from .model import candidates, save
from .registry import promotion_allowed, write_manifest


def load_params(path: str = "params.yaml") -> dict:
    config = Path(path)
    return yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}


def train(data_path: str, output_dir: str = "artifacts") -> dict:
    params = load_params()
    data_cfg = params.get("data", {})
    test_size = float(data_cfg.get("test_size", 0.20))
    random_state = int(data_cfg.get("random_state", 42))
    frame = load_dataset(data_path)
    validate(frame)
    y = (frame[TARGET].astype(str).str.lower() == "yes").astype(int)
    x = frame.drop(columns=[TARGET])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    results = {}
    with mlflow.start_run(run_name="bankpulse-training"):
        mlflow.log_params({"features": len(x.columns), "test_size": test_size, "random_state": random_state})
        for name, model in candidates(random_state).items():
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            threshold = select_threshold(y_test, probabilities)
            metrics = evaluate(y_test, probabilities, threshold)
            results[name] = metrics
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
        champion = max(results, key=lambda name: results[name]["roc_auc"])
        champion_model = candidates(random_state)[champion].fit(x_train, y_train)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifact = out / "model.joblib"
        save(champion_model, artifact)
        metrics = results[champion]
        write_manifest(artifact, metrics, out / "model.json")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.sklearn.log_model(champion_model, "model", registered_model_name="BankPulse")
        mlflow.log_artifact(str(artifact))
        mlflow.log_param("champion", champion)
        mlflow.log_param("promotion_allowed", promotion_allowed(metrics))
    return {"champion": champion, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
