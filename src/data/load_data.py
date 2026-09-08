import os
import hashlib
import json
from datetime import datetime

import pandas as pd
import yaml


def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    return pd.read_csv(file_path)


def compute_hash(df):
    """Compute a deterministic hash of a DataFrame for versioning."""
    csv_payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.md5(csv_payload).hexdigest()


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = y if y.nunique(dropna=False) > 1 else None
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )


def create_data_version(df, version_id, description, output_dir="data_versions"):
    """Create or update metadata for a versioned data snapshot."""
    os.makedirs(output_dir, exist_ok=True)

    snapshot_name = f"{version_id}.csv"
    snapshot_path = os.path.join(output_dir, snapshot_name)
    df.to_csv(snapshot_path, index=False)

    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
            metadata = loaded if isinstance(loaded, list) else [loaded]

    entry = {
        "version_id": version_id,
        "hash": compute_hash(df),
        "description": description,
        "n_rows": int(len(df)),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    metadata = [item for item in metadata if item.get("version_id") != version_id]
    metadata.append(entry)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return entry
