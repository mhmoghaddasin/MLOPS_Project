"""
Monitoring & Drift Detection Module
TODO: Implement the functions below

NOTE: Do NOT store sensitive information in logs.
"""
import json
import os
from datetime import datetime

import numpy as np
from scipy import stats


def log_prediction(prediction, confidence, latency, log_path="experiments/logs"):
    """Log a prediction with structured JSON format.
    TODO: Implement this function
    - Create a log entry with: timestamp, latency, prediction, confidence
    - Append to a daily log file (predictions_YYYY-MM-DD.json)
    - Do NOT store sensitive info (user_id, raw features, etc.)
    """
    pass


def detect_drift_ks(reference_data, current_data, threshold=0.05):
    """Detect drift using Kolmogorov-Smirnov test.
    TODO: Implement this function
    - Return dict with: method, statistic, p_value, drift_detected
    """
    pass


def detect_drift_mean_shift(reference_data, current_data, threshold=0.1):
    """Detect drift using mean shift detection.
    TODO: Implement this function
    - Return dict with: method, reference_mean, current_mean, drift_detected
    """
    pass
