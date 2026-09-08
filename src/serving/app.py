import time
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.monitoring.drift_detection import log_prediction


app = FastAPI(title="ML Model Serving API", version="1.0.0")

MODEL = None
METADATA = None


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    latency: float


@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup."""
    global MODEL, METADATA
    model_path = Path("models") / "model.joblib"
    metadata_path = Path("models") / "metadata.json"

    if model_path.exists():
        MODEL = joblib.load(model_path)

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            METADATA = json.load(file)
    else:
        METADATA = {}


@app.get("/health")
def health_check():
    """Return service status and model version."""
    model_loaded = MODEL is not None
    return {
        "status": "ok" if model_loaded else "model_unavailable",
        "model_version": (METADATA or {}).get("model_version"),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Make a prediction and log non-sensitive prediction metadata."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    expected_features = (METADATA or {}).get("input_features", [])
    if expected_features and len(request.features) != len(expected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(expected_features)} features, got {len(request.features)}",
        )

    start_time = time.perf_counter()
    if expected_features:
        features = pd.DataFrame([request.features], columns=expected_features)
    else:
        features = np.asarray(request.features, dtype=float).reshape(1, -1)

    prediction = int(MODEL.predict(features)[0])
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
    elif hasattr(MODEL, "decision_function"):
        score = float(np.ravel(MODEL.decision_function(features))[0])
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    latency = time.perf_counter() - start_time
    log_prediction(prediction, confidence, latency)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        latency=latency,
    )
