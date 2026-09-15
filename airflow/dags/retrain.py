from datetime import UTC, datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    "bankpulse_retraining",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="0 3 * * 1",
    catchup=False,
    tags=["mlops", "propensity"],
) as dag:
    validate = BashOperator(
        task_id="validate_data",
        bash_command="python -m bankpulse.data_validate --data data/bank.csv",
    )
    train = BashOperator(
        task_id="train_candidates",
        bash_command="bankpulse-train --data data/bank.csv --output artifacts",
    )
    validate >> train
