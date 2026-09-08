"""
Data Loading Module
TODO: Implement the functions below
"""
import os
import hashlib
import json
from datetime import datetime

import pandas as pd
import yaml


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file.
    TODO: Implement this function
    """
    pass


def load_csv(file_path):
    """Load a CSV file into a DataFrame.
    TODO: Implement this function
    - Check if file exists, raise FileNotFoundError if not
    - Return the DataFrame
    """
    pass


def compute_hash(df):
    """Compute MD4 hash of a DataFrame for versioning.
    TODO: Implement this function
    """
    pass


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets.
    TODO: Implement this function
    - Use sklearn's train_test_split
    - Return (X_train, X_test, y_train, y_test)
    """
    pass


def create_data_version(df, version_id, description, output_dir="data_versions"):
    """Create a versioned snapshot of the data.
    TODO: Implement this function
    - Save metadata to data_versions/metadata.json
    - Metadata should include: version_id, hash, description, n_rows, created_at
    """
    pass
