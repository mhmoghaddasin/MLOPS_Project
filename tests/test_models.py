import pytest
import pandas as pd
import numpy as np

from src.models.train_model import evaluate_model, get_model, train_model


class TestGetModel:
    """Tests for model creation."""

    def test_random_forest(self):
        """Test Random Forest model creation."""
        model = get_model("random_forest", {"n_estimators": 5, "random_state": 42})

        assert model is not None
        assert model.n_estimators == 5

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError):
            get_model("invalid", {})


class TestTrainModel:
    """Tests for model training."""

    @staticmethod
    def sample_data():
        X = pd.DataFrame(
            {
                "feature_1": np.arange(20),
                "feature_2": np.arange(20) % 3,
            }
        )
        y = pd.Series([0, 1] * 10)
        return X, y

    def test_train_model(self):
        """Test model training completes."""
        X, y = self.sample_data()

        model = train_model(
            X,
            y,
            "random_forest",
            {"n_estimators": 5, "random_state": 42},
        )

        assert hasattr(model, "predict")


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_evaluate_returns_metrics(self):
        """Test evaluation returns accuracy and f1_score."""
        X = pd.DataFrame({"feature_1": np.arange(20), "feature_2": np.arange(20) % 3})
        y = pd.Series([0, 1] * 10)
        model = train_model(
            X,
            y,
            "random_forest",
            {"n_estimators": 5, "random_state": 42},
        )

        metrics = evaluate_model(model, X, y)

        assert set(metrics) == {"accuracy", "f1_score"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
