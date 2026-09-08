"""
Tests for Data Loading Module
TODO: Write pytest tests for the following cases:
"""
import pytest
import pandas as pd
import numpy as np

# TODO: Import your functions
# from src.data.load_data import load_csv, split_data, compute_hash


class TestLoadData:
    """TODO: Implement tests for data loading."""

    def test_load_csv_existing_file(self):
        """Test loading an existing CSV file.
        TODO: Create a temp CSV, load it, check shape
        """
        pass

    def test_load_csv_missing_file(self):
        """Test that FileNotFoundError is raised for missing file.
        TODO: Call load_csv with nonexistent file, expect exception
        """
        pass

    def test_compute_hash(self):
        """Test hash computation is deterministic.
        TODO: Compute hash twice on same data, check they match
        """
        pass


class TestSplitData:
    """TODO: Implement tests for data splitting."""

    def test_split_returns_four_elements(self):
        """Test that split_data returns (X_train, X_test, y_train, y_test).
        TODO: Split sample data, check len(result) == 4
        """
        pass

    def test_split_sizes(self):
        """Test that split sizes are correct with test_size=0.2.
        TODO: Check train has 80% and test has 20% of samples
        """
        pass

    def test_split_reproducible(self):
        """Test that splitting is reproducible with same random_state.
        TODO: Split twice with same random_state, check results match
        """
        pass
