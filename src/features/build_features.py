"""
Feature Engineering Module
TODO: Implement the functions below
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif


def scale_features(X_train, X_test, method="standard"):
    """Scale features using the specified method.
    TODO: Implement this function
    - method: "standard", "minmax", or "none"
    - Return (X_train_scaled, X_test_scaled, scaler)
    """
    pass


def select_features(X_train, y_train, X_test, k=10):
    """Select top-k features using ANOVA F-value.
    TODO: Implement this function
    - Return (X_train_selected, X_test_selected, selector)
    """
    pass
