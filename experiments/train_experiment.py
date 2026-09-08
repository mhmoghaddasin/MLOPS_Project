"""Read config and run the full MLOps training pipeline."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import create_data_version, load_config, load_csv, split_data
from src.features.build_features import scale_features, select_features
from src.models.train_model import evaluate_model, save_model, train_model
from src.monitoring.drift_detection import build_drift_report


def _project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dataset_path(config, file_name):
    raw_path = _project_path(config["data"]["raw_path"])
    return raw_path if raw_path.suffix.lower() == ".csv" else raw_path / file_name


def _train_once(df, config, model_type, params):
    target_column = config["data"]["target_column"]
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_column,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
        method=config["features"]["scaling"],
    )
    X_train_selected, X_test_selected, selector = select_features(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=config["features"]["feature_selection_k"],
    )
    model = train_model(X_train_selected, y_train, model_type, params)
    metrics = evaluate_model(model, X_test_selected, y_test)

    pipeline = Pipeline(
        [
            ("scaler", scaler if scaler is not None else "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "input_features": X_train.columns.tolist(),
        "selected_features": X_train_selected.columns.tolist(),
    }


def _configure_mlflow(tracking_uri, experiment_name):
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        if db_path and not Path(db_path).is_absolute():
            tracking_uri = "sqlite:///" + _project_path(db_path).as_posix()
        mlflow.set_tracking_uri(tracking_uri)
    elif tracking_uri.startswith(("http://", "https://", "file:")):
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(str(_project_path(tracking_uri)))
    mlflow.set_experiment(experiment_name)


def run_pipeline(config_path="config/config.yaml"):
    """Run data versioning, MLflow experiments, drift report, and model saving."""
    config = load_config(_project_path(config_path))

    train_file = config["data"]["train_file"]
    drift_file = config["data"]["drift_file"]
    train_df = load_csv(_dataset_path(config, train_file))
    drift_df = load_csv(_dataset_path(config, drift_file))

    data_versions_dir = _project_path("data_versions")
    create_data_version(
        train_df,
        "v1",
        "Reference training data from dataset_v1.csv",
        output_dir=data_versions_dir,
    )
    create_data_version(
        drift_df,
        "v2",
        "Second data version from dataset_v2.csv with shifted feature distributions",
        output_dir=data_versions_dir,
    )

    _configure_mlflow(
        config["experiments"]["mlflow_tracking_uri"],
        config["experiments"]["experiment_name"],
    )

    results = []
    final_result = None
    run_configs = config["experiments"].get("runs", [])
    for run_config in run_configs:
        run_name = run_config["name"]
        data_version = run_config["data_version"]
        data_file = run_config["data_file"]
        model_type = run_config["model_type"]
        params = run_config.get("params", {})
        df = train_df if data_file == train_file else load_csv(_dataset_path(config, data_file))

        trained = _train_once(df, config, model_type, params)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("scaling", config["features"]["scaling"])
            mlflow.log_param("feature_selection_k", config["features"]["feature_selection_k"])
            mlflow.log_params(params)
            mlflow.log_metrics(trained["metrics"])
            mlflow.sklearn.log_model(
                trained["pipeline"],
                name="model",
                skops_trusted_types=[
                    "sklearn.feature_selection._univariate_selection.f_classif"
                ],
            )

        result = {
            "run_name": run_name,
            "data_version": data_version,
            "model_type": model_type,
            "metrics": trained["metrics"],
            "input_features": trained["input_features"],
            "selected_features": trained["selected_features"],
            "pipeline": trained["pipeline"],
        }
        results.append(result)

        if data_version == "v1" and model_type == config["model"]["type"] and final_result is None:
            final_result = result

    if final_result is None:
        trained = _train_once(
            train_df,
            config,
            config["model"]["type"],
            config["model"].get("params", {}),
        )
        final_result = {
            "run_name": "final_model",
            "data_version": "v1",
            "model_type": config["model"]["type"],
            "metrics": trained["metrics"],
            "input_features": trained["input_features"],
            "selected_features": trained["selected_features"],
            "pipeline": trained["pipeline"],
        }

    model_info = save_model(
        final_result["pipeline"],
        final_result["metrics"],
        data_version=final_result["data_version"],
        model_version="model_v1",
        output_dir=_project_path("models"),
        extra_metadata={
            "model_type": final_result["model_type"],
            "input_features": final_result["input_features"],
            "selected_features": final_result["selected_features"],
            "preprocessing": {
                "scaling": config["features"]["scaling"],
                "feature_selection_k": config["features"]["feature_selection_k"],
            },
        },
    )

    drift_report = build_drift_report(
        train_df,
        drift_df,
        features=config["monitoring"]["drift_features"],
        ks_threshold=config["monitoring"]["drift_threshold"],
        mean_shift_threshold=config["monitoring"]["mean_shift_threshold"],
    )
    drift_report_path = _project_path("experiments") / "drift_report.json"
    with open(drift_report_path, "w", encoding="utf-8") as file:
        json.dump(drift_report, file, indent=2)

    printable_results = [
        {key: value for key, value in result.items() if key != "pipeline"}
        for result in results
    ]
    summary = {
        "runs": printable_results,
        "saved_model": model_info,
        "drift_report_path": str(drift_report_path),
        "drift_detected": drift_report["drift_detected"],
        "affected_features": drift_report["affected_features"],
    }

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_pipeline()
