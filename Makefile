.PHONY: install lint test train serve batch dvc

install:
	python -m pip install -e '.[dev,mlops]'

lint:
	ruff check src tests

test:
	pytest -q

train:
	bankpulse-train --data Portugal-bank_marketing/Dataset/bank/bank.csv --output artifacts

serve:
	uvicorn bankpulse.api:app --host 0.0.0.0 --port 8000

batch:
	bankpulse-batch --data Portugal-bank_marketing/Dataset/bank/bank.csv --model artifacts/model.joblib --output artifacts/scored.parquet

dvc:
	dvc repro
