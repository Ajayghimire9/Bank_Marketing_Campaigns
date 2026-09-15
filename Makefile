.PHONY: install lint test train serve batch dvc

install:
	python -m pip install -e '.[dev,mlops]'

lint:
	ruff check src tests

test:
	pytest -q

train:
	bankpulse-train --data Portugal-bank_marketing/Dataset/bank-additional/bank-additional/bank-additional.csv --output artifacts

serve:
	uvicorn bankpulse.api:app --host 0.0.0.0 --port 8000

batch:
	bankpulse-batch --data Portugal-bank_marketing/Dataset/bank-additional/bank-additional/bank-additional.csv --model artifacts/model.joblib --output artifacts/scored.parquet

dvc:
	dvc repro
