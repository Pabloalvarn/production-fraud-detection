# Executive summary

## Decision

The project evaluates whether a calibrated machine-learning model can prioritise rare fraudulent PaySim transactions more effectively than dummy and rule-based baselines while respecting review costs and capacity.

## Recommended model

- Model: XGBoost
- Calibration: sigmoid
- Medium-risk threshold: 0.86
- High-risk threshold: 0.90
- Review rate at the cost-minimising threshold: 1.4877%

## Test performance

| Measure | Result |
|---|---:|
| Fraud prevalence | 1.3994% |
| Average Precision | 0.992922 |
| ROC-AUC | 0.999905 |
| Precision | 0.939144 |
| Recall | 0.998403 |
| F1 | 0.967867 |
| Matthews correlation | 0.967863 |
| Fraud-value capture | 99.9763% |
| Brier score | 0.000936 |

## Business scenario

At the selected validation threshold, the locked test policy:

- Reviews 1.4877% of transactions.
- Identifies 99.8403% of fraudulent transactions.
- Captures 99.9763% of simulated fraudulent value.
- Produces a scenario-based cost of 413,233.61 cost units.
- Produces 81 false positives and two false negatives across 89,466 test transactions.

These figures are based on synthetic transactions and explicit cost assumptions. They are not forecasts of real financial savings.

## Recommendation

Select calibrated XGBoost as the portfolio candidate. It materially outperformed Logistic Regression: Average Precision increased from 0.8043 to 0.9929, precision from 0.2615 to 0.9391 and F1 from 0.4128 to 0.9679, while the scenario-based cost decreased from 859,389.44 to 413,233.61.

The cost-minimising threshold and the capacity-constrained policy should be treated as distinct operational scenarios. The selected threshold reviews 1.49% of transactions and detects 99.84% of fraud. If review capacity is strictly limited to 1%, the model achieves 99.89% precision but recall falls to 71.41%. A real deployment would require the capacity and cost assumptions to be agreed by fraud operations before selecting an operating point.

Permutation importance indicates that origin balance and transaction amount dominate predictions in PaySim. This should be treated as a property of the synthetic data-generating process, not evidence of real-world causal drivers.

## Principal risks

- Synthetic patterns may overstate model performance.
- Business costs are illustrative.
- Labels and feature availability differ from real payment systems.
- No protected attributes are available for a fairness assessment.
- Real deployment would require security, compliance and model-risk approval.

