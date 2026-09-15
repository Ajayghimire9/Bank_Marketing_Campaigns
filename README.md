# BankPulse

Marketing propensity experiments and model serving.

BankPulse estimates customer response propensity from the bank-marketing dataset. It connects data validation, candidate comparison, a versioned model artifact and batch or API scoring.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m bankpulse.train --data Portugal-bank_marketing/Dataset/bank-additional/bank-additional/bank-additional.csv
uvicorn bankpulse.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

The default split is 60% training, 20% validation and 20% test. Candidate selection and threshold tuning use validation data; the selected model is measured once on the test set.

Call duration is excluded because it is unknown before the customer is contacted. Both yes/no and 0/1 targets follow the same mapping.

The API and batch scorer read the selected threshold from model.json and verify the model checksum before loading it. Batch input does not need target labels.

MLflow records candidate experiments; DVC and Airflow describe training stages. Kubernetes manifests and Prometheus configuration are included as deployment assets.

## Technology

Python, scikit-learn, MLflow, DVC, FastAPI, Parquet, Airflow, Docker, Kubernetes, Prometheus.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The random split is an IID experiment, not a temporal campaign simulation. Thresholds and calibration need revalidation for a new population. A checksum detects changed bytes but is not a signature; load only artifacts from a trusted source. No campaign revenue or production deployment is claimed.
