from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from .data import FEATURES
from .model import load
from .schema import HealthResponse, PredictionRequest, PredictionResponse

MODEL_PATH = Path(os.getenv("BANKPULSE_MODEL", "artifacts/model.joblib"))
REQUESTS = Counter("bankpulse_prediction_requests_total", "Prediction requests")
LATENCY = Histogram("bankpulse_prediction_latency_seconds", "Prediction latency")
app = FastAPI(title="BankPulse Inference API", version="1.0.0")
app.mount("/metrics", make_asgi_app())
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model artifact unavailable")
        _model = load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=MODEL_PATH.exists())


@app.get("/ready")
def ready():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = time.perf_counter()
    REQUESTS.inc()
    frame = pd.DataFrame([request.model_dump()], columns=FEATURES)
    probability = float(get_model().predict_proba(frame)[0, 1])
    latency = time.perf_counter() - started
    LATENCY.observe(latency)
    return PredictionResponse(
        probability=probability,
        prediction=int(probability >= 0.5),
        model_version="bankpulse-1.0.0",
        latency_ms=latency * 1000,
    )
