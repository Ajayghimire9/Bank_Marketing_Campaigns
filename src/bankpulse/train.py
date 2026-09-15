from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split

from .data import TARGET, load_dataset, validate
from .evaluate import evaluate, select_threshold
from .model import candidates, save
from .registry import promotion_allowed, write_manifest


def train(data_path: str, output_dir: str = "artifacts") -> dict:
    frame = load_dataset(data_path)
    validate(frame)
    y = (frame[TARGET].astype(str).str.lower() == "yes").astype(int)
    x = frame.drop(columns=[TARGET])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri("file:./mlruns")
    results = {}
    with mlflow.start_run(run_name="bankpulse-training"):
        mlflow.log_param("features", len(x.columns))
        for name, model in candidates().items():
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            threshold = select_threshold(y_test, probabilities)
            metrics = evaluate(y_test, probabilities, threshold)
            results[name] = metrics
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
        champion = max(results, key=lambda name: results[name]["roc_auc"])
        model = candidates()[champion].fit(x_train, y_train)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifact = out / "model.joblib"
        save(model, artifact)
        manifest = out / "model.json"
        write_manifest(artifact, results[champion], manifest)
        mlflow.log_artifact(str(artifact))
        mlflow.log_param("champion", champion)
        mlflow.log_param("promotion_allowed", promotion_allowed(results[champion]))
    return {"champion": champion, "metrics": results[champion]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
