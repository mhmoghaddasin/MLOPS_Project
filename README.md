# MLOPS Project - Customer Churn Prediction

## Student Info
- Name: MohammadHadi Moghaddasin
- Student ID:

## Project Description
This project implements an end-to-end MLOps workflow for a binary customer churn classification problem. The target column is `churn`, where `0` means the customer stayed and `1` means the customer churned.

The workflow includes:
- Modular project structure
- Reproducible configuration through `config/config.yaml`
- Data loading and data version metadata
- Feature scaling and feature selection with `SelectKBest(f_classif)`
- Model training and evaluation
- MLflow experiment tracking with at least three runs
- Model artifact and metadata saving
- Monitoring logs for predictions
- Drift detection between `dataset_v1.csv` and `dataset_v2.csv`
- FastAPI model serving
- Docker packaging
- GitHub Actions CI
- Pytest tests

## Dataset
The assignment data files must be placed one directory above this project:

```text
MLOPS_Project/
├── dataset_v1.csv
├── dataset_v2.csv
└── Project/
```

The project expects these paths:

```text
../dataset_v1.csv
../dataset_v2.csv
```

Dataset usage:
- `dataset_v1.csv`: reference training data and final model training source
- `dataset_v2.csv`: second data version for data versioning and drift detection

Dataset address:
- `dataset_v1.csv`: https://drive.google.com/file/d/1dF_84zIvCUnTd3BH2XLUGfNby0hmgXQC/view?usp=sharing
- `dataset_v2.csv`: https://drive.google.com/file/d/1jwB1nBY0CL2lFD51IvP4wBOnNRrDq-V6/view?usp=sharing

### Dataset Features

Both datasets contain the same 12 feature columns and the target column `churn`.

| Feature | Description |
|---|---|
| `months_tenure` | Customer tenure in months. |
| `charges_monthly` | Customer's monthly bill amount. |
| `charges_total` | Total amount paid by the customer to date. |
| `services_num` | Number of active services. |
| `tickets_support` | Number of support tickets submitted by the customer. |
| `minutes_session_avg` | Average duration of each usage session. |
| `frequency_login` | Number of logins per month. |
| `length_contract` | Contract length in months; values are 1, 12, or 24. |
| `days_delay_payment` | Number of days the customer is late in making payments. |
| `score_credit` | Customer credit score. |
| `age` | Customer age. |
| `dependents_num` | Number of dependents. |
| `churn` | Target variable: `0` means the customer stayed and `1` means the customer churned. |

## Project Structure
```text
Project/
├── .github/workflows/ci.yml
├── config/config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── data_versions/
│   └── metadata.json
├── experiments/
│   ├── drift_report.json
│   └── train_experiment.py
├── models/
│   ├── metadata.json
│   └── model.joblib
├── src/
│   ├── data/load_data.py
│   ├── features/build_features.py
│   ├── models/train_model.py
│   ├── monitoring/drift_detection.py
│   └── serving/app.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup
Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `mlflow`, `pytest`, or `uvicorn` are not recognized in PowerShell, run them through the virtualenv Python:

```powershell
.\venv\Scripts\python.exe -m mlflow --version
.\venv\Scripts\python.exe -m pytest --version
```

## Configuration
All important parameters are stored in `config/config.yaml`, not hardcoded in the training code:

```yaml
data:
  raw_path: "../"
  train_file: "dataset_v1.csv"
  drift_file: "dataset_v2.csv"
  target_column: "churn"
  test_size: 0.2
  random_state: 42

features:
  scaling: "standard"
  feature_selection_k: 10

model:
  type: "logistic_regression"
  params:
    C: 1.0
    max_iter: 1000
    random_state: 42
    class_weight: "balanced"
```

The `model` section defines the final model that is saved to `models/model.joblib`.

The `experiments.runs` section defines the MLflow comparison runs. It includes different data versions, model types, and hyperparameters.

## Run Training Pipeline
```powershell
.\venv\Scripts\python.exe experiments\train_experiment.py
```

The training pipeline:
- reads `config/config.yaml`
- loads `dataset_v1.csv` and `dataset_v2.csv`
- creates `v1` and `v2` metadata in `data_versions/metadata.json`
- runs three MLflow experiments
- logs parameters, accuracy, F1 score, and model artifacts
- trains the final model on `dataset_v1.csv`
- saves `models/model.joblib`
- saves `models/metadata.json`
- creates `experiments/drift_report.json`

## MLflow
This project uses a local SQLite MLflow backend:

```yaml
mlflow_tracking_uri: "sqlite:///experiments/mlflow.db"
```

This was used instead of `http://localhost:5000` because `http://localhost:5000` requires a separate MLflow tracking server to be running before the pipeline starts. The SQLite backend lets the homework run locally with one command.

Start MLflow UI with:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

If you prefer the original server style, first start a tracking server:

```powershell
.\venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///experiments/mlflow.db --host 127.0.0.1 --port 5000
```

Then set this in `config/config.yaml`:

```yaml
experiments:
  mlflow_tracking_uri: "http://localhost:5000"
```

## Tests
Run all tests:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Test coverage includes:
- loading an existing CSV file
- raising `FileNotFoundError` for a missing CSV file
- deterministic data hash generation
- train/test split output and reproducibility
- feature selection shape
- model creation
- invalid model type handling
- model training
- model evaluation metrics
- KS drift detection

## Monitoring
Prediction monitoring is implemented in `src/monitoring/drift_detection.py`.

Each prediction log contains:

```json
{
  "timestamp": "...",
  "latency": 0.023,
  "prediction": 1,
  "confidence": 0.87
}
```

Sensitive data such as user IDs and raw features are not logged.

## Drift Detection
Drift detection compares `dataset_v1.csv` as reference data with `dataset_v2.csv` as current data.

Configured drift features:
- `monthly_charges`
- `support_tickets`
- `credit_score`

Implemented methods:
- Kolmogorov-Smirnov Test
- Mean Shift Detection

The drift report is saved to:

```text
experiments/drift_report.json
```

## Serve the Model
Train the model first:

```powershell
.\venv\Scripts\python.exe experiments\train_experiment.py
```

Start the API:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

Prediction example:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"features\":[22.85,74.43,1570.81,1,2,28.11,12.89,12,8.43,698,46,3]}"
```

## Docker
Docker Hub image:

```text
https://hub.docker.com/r/moghaddasin1366/mlops_project
```

Build the Docker image:

```powershell
docker build -t mlops-homework-01 .
```

Run the local image:

```powershell
docker run -p 8000:8000 mlops-homework-01
```

Pull and run the Docker Hub image:

```powershell
docker pull moghaddasin1366/mlops_project:latest
docker run -p 8000:8000 moghaddasin1366/mlops_project:latest
```

Open:

```text
http://localhost:8000/health
```

## CI
GitHub Actions CI is defined in `.github/workflows/ci.yml`.

It runs on:
- `push`
- `pull_request`

CI steps:
- checkout repository
- install Python 3.11
- install dependencies
- run `pytest tests/ -v`

## Git
The project includes meaningful commits for:
- training, data versioning, MLflow, and drift pipeline
- serving, Docker, CI, and documentation
- pytest coverage

Check commit history:

```powershell
git log --oneline
```
