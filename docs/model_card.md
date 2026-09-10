# Model card — calibrated XGBoost fraud classifier

## Model details

- Name: PaySim pre-authorisation fraud classifier
- Version: 0.1.0
- Owner: Pablo Álvarez Arnedo
- Model family: XGBoost with sigmoid probability calibration
- Training code: `src/fraud_detection/models/train.py`
- Model artifact: `artifacts/model_bundle.joblib` (not committed)

## Intended use

Portfolio demonstration of a risk-ranking system for rare fraudulent transactions. The output is a probability used by a configurable decision policy.

## Prohibited use

This model is trained on synthetic data. It must not approve, reject, block or investigate real transactions and must not be treated as validated financial software.

## Inputs

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`

## Excluded inputs

- Target and policy outputs: `isFraud`, `isFlaggedFraud`
- Post-transaction variables: `newbalanceOrig`, `newbalanceDest`
- Raw entity identifiers: `nameOrig`, `nameDest`

## Validation design

- Chronological train/validation/test split by whole `step` values.
- Earlier validation half: probability calibration.
- Later validation half: model threshold.
- Final test: one locked evaluation.

## Evaluation results

| Metric | Result |
|---|---:|
| Fraud prevalence | 1.3994% |
| Average Precision | 0.992922 |
| ROC-AUC | 0.999905 |
| Precision | 0.939144 |
| Recall | 0.998403 |
| F1 | 0.967867 |
| F2 | 0.985960 |
| Matthews correlation | 0.967863 |
| Brier score | 0.000936 |
| Log loss | 0.003947 |
| Review rate | 1.4877% |
| Fraud-value capture rate | 99.9763% |
| Scenario-based total cost | 413,233.61 |

At the selected threshold, the locked test confusion matrix contained 88,133 true negatives, 81 false positives, two false negatives and 1,250 true positives.

For comparison, Logistic Regression achieved 0.804296 Average Precision, 0.261453 precision, 0.980032 recall and an estimated scenario cost of 859,389.44.

Under a separate policy that reviews only the highest-scored 1% of test transactions, XGBoost achieved 99.8883% precision and 71.4058% recall. This capacity-constrained policy is reported separately from the cost-minimising operating point.

## Threshold

- Medium-risk threshold: 0.86
- High-risk threshold: 0.90
- Selection data: later validation partition
- Objective: minimum assumed business cost

## Explainability

Model-agnostic permutation importance was computed using Average Precision on the locked chronological test set with three repeats. `oldbalanceOrg` (0.9262) and `amount` (0.9237) were the dominant raw inputs, followed by `type` (0.1525), `oldbalanceDest` (0.0503) and `step` (0.0000). Importance values describe dependence learned from synthetic patterns and do not establish causal effects.

## Limitations and risks

- Synthetic dataset and synthetic fraud mechanisms.
- Cost parameters are assumptions.
- No protected attributes for fairness assessment.
- Possible temporal drift.
- Delayed-label monitoring is simulated.
- No external or real-world validation.
- The cost-minimising policy reviews 1.49% of transactions and therefore exceeds the separately assumed 1% capacity constraint.

## Monitoring

Monitor schema, missingness, category changes, feature drift, score drift, review rate and, when delayed labels arrive, Average Precision, recall, captured fraud value and calibration.

## Approval status

Portfolio demonstration only — not approved for production use.

