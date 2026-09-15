from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from .data import FEATURES
from .model import load
from .registry import serving_threshold
from .schema import HealthResponse, PredictionRequest, PredictionResponse

MODEL_PATH = Path(os.getenv("BANKPULSE_MODEL", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("BANKPULSE_MODEL_VERSION", "bankpulse-1.0.0")
DECISION_THRESHOLD = None
REQUESTS = Counter("bankpulse_prediction_requests_total", "Prediction requests")
ERRORS = Counter("bankpulse_prediction_errors_total", "Prediction errors")
LATENCY = Histogram("bankpulse_prediction_latency_seconds", "Prediction latency")
app = FastAPI(title="BankPulse Inference API", version="2.0.0")
app.mount("/metrics", make_asgi_app())
_model = None


def get_model():
    global _model, DECISION_THRESHOLD
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model artifact unavailable")
        DECISION_THRESHOLD = serving_threshold(MODEL_PATH)
        _model = load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=MODEL_PATH.exists())


@app.get("/ready")
def ready():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        get_model()
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Model validation failed") from exc
    return {"ready": True, "model_version": MODEL_VERSION}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = time.perf_counter()
    REQUESTS.inc()
    try:
        frame = pd.DataFrame([request.model_dump()], columns=FEATURES)
        probability = float(get_model().predict_proba(frame)[0, 1])
    except HTTPException:
        ERRORS.inc()
        raise
    except Exception as exc:
        ERRORS.inc()
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    elapsed = time.perf_counter() - started
    LATENCY.observe(elapsed)
    return PredictionResponse(
        probability=probability,
        prediction=int(probability >= DECISION_THRESHOLD),
        model_version=MODEL_VERSION,
        latency_ms=elapsed * 1000,
    )
