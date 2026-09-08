import os
import json
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def get_model(model_type, params):
    """Get a model instance based on type and parameters."""
    params = dict(params or {})
    model_type = (model_type or "").lower()

    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    if model_type == "logistic_regression":
        params.setdefault("max_iter", 1000)
        return LogisticRegression(**params)

    if model_type == "svm":
        params.setdefault("probability", True)
        return SVC(**params)

    raise ValueError(
        'model_type must be one of "random_forest", "logistic_regression", or "svm"'
    )


def train_model(X_train, y_train, model_type, params):
    """Train a model and return the fitted estimator."""
    model = get_model(model_type, params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model with accuracy and F1."""
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }


def save_model(
    model,
    metrics,
    data_version,
    model_version,
    output_dir="models",
    extra_metadata=None,
):
    """Save a model artifact and its metadata."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    metadata_path = os.path.join(output_dir, "metadata.json")
    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "data_version": data_version,
        "metric": dict(metrics),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {"model_path": model_path, "metadata_path": metadata_path, "metadata": metadata}
