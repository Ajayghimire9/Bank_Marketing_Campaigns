from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES

NUMERIC = [
    "age", "campaign", "pdays", "previous", "emp_var_rate", "cons_price_idx",
    "cons_conf_idx", "euribor3m", "nr_employed",
]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


def build_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("features", prep), ("model", model)])
