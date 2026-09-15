# BankPulse — Enterprise-Style Marketing Propensity MLOps

BankPulse transforms the original bank-marketing classification exercise into a production-oriented machine-learning platform for campaign propensity scoring.

> **Portfolio scope:** this repository demonstrates ML engineering and MLOps patterns. It is not a production banking decision engine and does not claim regulatory approval or live financial performance.

## Architecture

```text
                         ┌──────────────────────┐
                         │ Versioned Bank Data  │
                         │ CSV / DVC            │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ Data Contract        │
                         │ Schema + Validation  │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ Feature Pipeline     │
                         │ Imputation + OHE     │
                         │ Leakage Controls     │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     v                             v
              Logistic Regression          Random Forest / HGB
                     │                             │
                     └──────────────┬──────────────┘
                                    v
                         ┌──────────────────────┐
                         │ Evaluation Gate      │
                         │ ROC-AUC / PR-AUC     │
                         │ F1 / Recall / ECE    │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ MLflow               │
                         │ Experiments          │
                         │ Model Registry       │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    v                                v
             Batch Scoring                       FastAPI
             Parquet Output                      Online API
                    │                                │
                    └───────────────┬────────────────┘
                                    v
                         ┌──────────────────────┐
                         │ Observability        │
                         │ Prometheus + Alerts  │
                         │ Drift + Calibration  │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ Airflow Retraining   │
                         │ Scheduled Pipeline   │
                         └──────────────────────┘

              Docker → Kubernetes → HPA
              GitHub Actions → CI → Artifacts
```

## Engineering capabilities

### Data engineering
- Explicit feature contract
- Dataset validation before training
- DVC pipeline definition
- Versioned training parameters
- Offline feature-store abstraction using Parquet
- Deterministic train/test split

### Machine learning
- Logistic Regression baseline
- Random Forest
- Histogram Gradient Boosting
- Class imbalance handling
- Pipeline-based preprocessing
- Leakage-aware feature selection
- Automated champion selection
- Optimised decision threshold
- ROC-AUC, PR-AUC, precision, recall and F1
- Probability calibration / ECE utility
- Permutation-based explainability
- Cost-aware campaign policy and top-cohort lift analysis

### MLOps
- MLflow experiment tracking
- MLflow model registration
- Model manifest
- SHA-256 artifact integrity
- Candidate/promotion gate
- Reproducible `params.yaml`
- DVC stages
- Airflow scheduled retraining
- CI quality gates
- CI training workflow with downloadable model artifacts

### Production serving
- FastAPI inference gateway
- Strict Pydantic request contracts
- `/health`
- `/ready`
- `/v1/predict`
- `/metrics`
- Configurable model version and decision threshold
- Lazy model loading
- Prometheus request/error/latency metrics
- Docker runtime
- Kubernetes Deployment + Service
- Kubernetes HPA
- Readiness/liveness probes
- Resource requests/limits
- Read-only filesystem / privilege restrictions

### Monitoring
- Request rate
- Error rate
- Latency histograms
- Prediction-rate monitoring
- Population Stability Index drift
- Calibration error
- Prometheus alert rules

## Important modelling decision

The original dataset contains `duration`, the length of the last call. BankPulse excludes it from the production feature contract because it is only known after a contact has taken place. Including it in a pre-contact targeting model would introduce post-event leakage and produce an unrealistic offline evaluation.

## Repository layout

```text
src/bankpulse/
├── api.py             # online inference
├── batch.py           # offline scoring
├── business.py        # cost-aware campaign policy
├── calibration.py     # probability calibration metrics
├── config.py          # typed settings
├── data.py            # loading and validation
├── data_validate.py   # DVC validation stage
├── drift.py           # PSI drift detection
├── evaluate.py        # model metrics and threshold search
├── explain.py         # permutation importance
├── feature_store.py   # offline feature-store abstraction
├── features.py        # preprocessing pipeline
├── model.py           # candidate model factory
├── monitoring.py      # serving metrics
├── registry.py        # artifact integrity / promotion gate
├── schema.py          # API contracts
└── train.py           # MLflow training pipeline

airflow/dags/retrain.py
k8s/deployment.yaml
k8s/hpa.yaml
monitoring/prometheus.yml
monitoring/alerts.yml
dvc.yaml
params.yaml
MODEL_CARD.md
Dockerfile
docker-compose.yml
Makefile
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mlops]'

make lint
make test
```

Train against the repository dataset:

```bash
make train
```

Start the API after a model artifact has been generated:

```bash
make serve
```

Run batch scoring:

```bash
make batch
```

Start the local MLflow + Prometheus stack:

```bash
docker compose up --build
```

## API contract

The prediction endpoint accepts the feature contract and returns:

```json
{
  "probability": 0.73,
  "prediction": 1,
  "model_version": "bankpulse-2.0.0",
  "latency_ms": 4.2
}
```

## Production lifecycle

```text
Data change
   ↓
DVC / validation
   ↓
Training candidates
   ↓
MLflow experiment
   ↓
Evaluation + calibration
   ↓
Promotion gate
   ↓
Model registry
   ↓
Batch / online deployment
   ↓
Prometheus monitoring
   ↓
Drift / quality signal
   ↓
Scheduled retraining
```

## Model governance

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, limitations, data leakage considerations, evaluation methodology and governance boundaries.

No credentials, cloud keys or secrets are stored in the repository. Cloud deployment can be connected through environment variables or a secret manager without changing the model-serving code.
