import pandas as pd
from sklearn.linear_model import LogisticRegression

from bankpulse.features import build_pipeline


def test_pipeline_fits_mixed_data():
    frame = pd.DataFrame(
        {
            "age": [30, 50],
            "job": ["admin", "services"],
            "marital": ["single", "married"],
            "education": ["high", "university"],
            "default": ["no", "no"],
            "housing": ["yes", "no"],
            "loan": ["no", "yes"],
            "contact": ["cellular", "telephone"],
            "month": ["may", "jun"],
            "day_of_week": ["mon", "fri"],
            "campaign": [1, 2],
            "pdays": [999, 10],
            "previous": [0, 1],
            "poutcome": ["nonexistent", "failure"],
            "emp_var_rate": [1.0, -1.0],
            "cons_price_idx": [93.0, 94.0],
            "cons_conf_idx": [-40.0, -35.0],
            "euribor3m": [4.5, 1.2],
            "nr_employed": [5000, 5100],
        }
    )
    model = build_pipeline(LogisticRegression(max_iter=200)).fit(frame, [0, 1])
    assert model.predict_proba(frame).shape == (2, 2)
