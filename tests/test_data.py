import pytest
import pandas as pd
import numpy as np

from src.data.load_data import compute_hash, load_csv, split_data


class TestLoadData:
    """Tests for data loading."""

    def test_load_csv_existing_file(self, tmp_path):
        """Test loading an existing CSV file."""
        csv_path = tmp_path / "sample.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

        df = load_csv(csv_path)

        assert df.shape == (2, 2)
        assert df.columns.tolist() == ["a", "b"]

    def test_load_csv_missing_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_csv("missing.csv")

    def test_compute_hash(self):
        """Test hash computation is deterministic."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        assert compute_hash(df) == compute_hash(df.copy())


class TestSplitData:
    """Tests for data splitting."""

    @staticmethod
    def sample_data(n_rows=10):
        return pd.DataFrame(
            {
                "feature_1": np.arange(n_rows),
                "feature_2": np.arange(n_rows) * 2,
                "churn": [0, 1] * (n_rows // 2),
            }
        )

    def test_split_returns_four_elements(self):
        """Test that split_data returns (X_train, X_test, y_train, y_test).
        """
        result = split_data(self.sample_data(), "churn", test_size=0.2, random_state=42)

        assert len(result) == 4

    def test_split_sizes(self):
        """Test that split sizes are correct with test_size=0.2."""
        X_train, X_test, y_train, y_test = split_data(
            self.sample_data(),
            "churn",
            test_size=0.2,
            random_state=42,
        )

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_split_reproducible(self):
        """Test that splitting is reproducible with same random_state."""
        first = split_data(self.sample_data(), "churn", test_size=0.2, random_state=42)
        second = split_data(self.sample_data(), "churn", test_size=0.2, random_state=42)

        for left, right in zip(first, second):
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right)
            else:
                pd.testing.assert_series_equal(left, right)
