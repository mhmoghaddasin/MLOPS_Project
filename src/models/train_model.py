"""
Model Training Module
TODO: Implement the functions below
"""
import os
import json
import joblib
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def get_model(model_type, params):
    """Get a model instance based on type and parameters.
    TODO: Implement this function
    - model_type: "random_forest", "logistic_regression", or "svm"
    - Return the initialized model
    """
    pass


def train_model(X_train, y_train, model_type, params):
    """Train a model.
    TODO: Implement this function
    - Return the trained model
    """
    pass


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model.
    TODO: Implement this function
    - Return dict with "accuracy" and "f1_score"
    """
    pass


def save_model(model, metrics, data_version, model_version, output_dir="models"):
    """Save model and its metadata.
    TODO: Implement this function
    - Save model as model.joblib
    - Save metadata.json with: model_version, created_at, data_version, metric
    """
    pass
