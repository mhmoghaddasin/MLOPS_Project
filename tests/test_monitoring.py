import numpy as np

from src.monitoring.drift_detection import detect_drift_ks


class TestDriftDetection:
    """Tests for drift detection."""

    def test_no_drift_same_distribution(self):
        """Test no drift when reference and current are the same."""
        data = np.linspace(0, 1, 100)

        result = detect_drift_ks(data, data, threshold=0.05)

        assert result["drift_detected"] is False

    def test_drift_different_distributions(self):
        """Test drift detected for shifted distribution."""
        reference = np.linspace(0, 1, 100)
        current = np.linspace(5, 6, 100)

        result = detect_drift_ks(reference, current, threshold=0.05)

        assert result["drift_detected"] is True

    def test_ks_result_structure(self):
        """Test that KS result has correct keys."""
        result = detect_drift_ks(np.arange(10), np.arange(10), threshold=0.05)

        assert set(result) == {"method", "statistic", "p_value", "drift_detected"}
