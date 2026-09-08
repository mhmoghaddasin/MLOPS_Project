import json
import os
from datetime import datetime

import numpy as np
from scipy import stats


def log_prediction(prediction, confidence, latency, log_path="experiments/logs"):
    """Log a prediction with structured JSON format.

    The log deliberately excludes raw input features and user identifiers.
    """
    os.makedirs(log_path, exist_ok=True)
    now = datetime.utcnow()
    entry = {
        "timestamp": now.isoformat(timespec="milliseconds") + "Z",
        "latency": float(latency),
        "prediction": int(prediction),
        "confidence": float(confidence),
    }

    file_path = os.path.join(log_path, f"predictions_{now.date().isoformat()}.json")
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return entry


def detect_drift_ks(reference_data, current_data, threshold=0.05):
    """Detect drift using the Kolmogorov-Smirnov test."""
    reference = np.asarray(reference_data, dtype=float).ravel()
    current = np.asarray(current_data, dtype=float).ravel()

    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "method": "ks_test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold),
    }


def detect_drift_mean_shift(reference_data, current_data, threshold=0.1):
    """Detect drift using normalized mean shift."""
    reference = np.asarray(reference_data, dtype=float).ravel()
    current = np.asarray(current_data, dtype=float).ravel()

    reference_mean = float(np.mean(reference))
    current_mean = float(np.mean(current))
    denominator = max(abs(reference_mean), 1e-12)
    drift_score = abs(current_mean - reference_mean) / denominator

    return {
        "method": "mean_shift",
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "drift_score": float(drift_score),
        "drift_detected": bool(drift_score > threshold),
    }


def build_drift_report(
    reference_df,
    current_df,
    features,
    ks_threshold=0.05,
    mean_shift_threshold=0.1,
):
    """Build a per-feature drift report for monitoring."""
    feature_reports = {}
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ks_result = detect_drift_ks(
            reference_df[feature],
            current_df[feature],
            threshold=ks_threshold,
        )
        mean_shift_result = detect_drift_mean_shift(
            reference_df[feature],
            current_df[feature],
            threshold=mean_shift_threshold,
        )
        feature_reports[feature] = {
            "ks": ks_result,
            "mean_shift": mean_shift_result,
            "drift_detected": bool(
                ks_result["drift_detected"] or mean_shift_result["drift_detected"]
            ),
            "severity": float(
                max(ks_result["statistic"], mean_shift_result["drift_score"])
            ),
        }

    affected_features = [
        feature
        for feature, result in feature_reports.items()
        if result["drift_detected"]
    ]
    drift_detected = bool(affected_features)

    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "drift_detected": drift_detected,
        "affected_features": affected_features,
        "features": feature_reports,
        "recommended_action": (
            "Review feature distributions and consider retraining with dataset_v2.csv."
            if drift_detected
            else "Continue monitoring."
        ),
    }
