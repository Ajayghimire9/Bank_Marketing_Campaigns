# BankPulse Model Card

## Intended use

Propensity modelling for prioritising bank-marketing outreach in a portfolio engineering exercise.

## Data

The project uses the original bank-marketing dataset already stored in this repository. The production feature contract intentionally excludes `duration` because it is only known after a contact and would leak post-contact information into a pre-contact targeting model.

## Training

Candidates are trained behind a single preprocessing pipeline. Logistic Regression provides an interpretable baseline; Random Forest and HistGradientBoosting provide non-linear alternatives. The champion is selected on ROC-AUC, while threshold selection is evaluated separately.

## Evaluation

Tracked metrics include ROC-AUC, PR-AUC, precision, recall, F1 and the selected decision threshold. Probability calibration and population drift utilities are included for production monitoring.

## Governance

Every exported artifact receives a model manifest containing version, metrics and SHA-256 checksum. MLflow records experiment parameters, metrics and the registered model artifact. Promotion is gated by configurable quality thresholds.

## Limitations

This is a portfolio project, not a live banking decision system. Fairness, privacy, regulatory compliance, human review, causal uplift modelling and real-time feature freshness require additional production controls and institution-specific validation.
