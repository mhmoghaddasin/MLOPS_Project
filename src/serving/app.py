"""
Model Serving Module
TODO: Implement the endpoints below
"""
import time
import joblib
import json
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


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
    """Load model on startup.
    TODO: Load model.joblib and metadata.json from models/ directory
    """
    global MODEL, METADATA
    pass


@app.get("/health")
def health_check():
    """Health check endpoint.
    TODO: Return status and model_version
    """
    pass


@app.post("/predict")
def predict(request: PredictionRequest):
    """Make a prediction.
    TODO: Implement prediction logic
    - Use MODEL.predict() and MODEL.predict_proba()
    - Log the prediction using log_prediction()
    - Return PredictionResponse
    """
    pass
