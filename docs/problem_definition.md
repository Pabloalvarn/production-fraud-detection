# Problem definition

## Business context

A payment provider processes a large volume of mobile-money transactions. Fraud is rare, so approving every transaction produces high apparent accuracy but unacceptable losses. Investigating every transaction is also impossible because reviews cost money, delay customers and exceed operational capacity.

The system must rank transactions by risk and support one of three actions:

1. **Approve** low-risk transactions.
2. **Review** medium-risk transactions manually.
3. **Request enhanced verification** for the highest-risk transactions.

The model is a decision-support component, not an autonomous production control.

## Predictive task

For a transaction with features available before authorisation, estimate:

```text
P(transaction is fraudulent | information available at decision time)
```

The target is `isFraud`. The primary use case excludes post-transaction balances, raw account identifiers and `isFlaggedFraud`.

## Stakeholders

- Fraud operations: prioritises manual investigations.
- Risk management: sets loss assumptions and risk appetite.
- Customer operations: monitors false positives and delays.
- Data science: develops and validates the model.
- ML engineering: serves and monitors predictions.
- Compliance/model risk: reviews data, explanations and limitations.

## Success criteria

The portfolio project is successful when it:

- Beats dummy and rule-based baselines on temporal holdout data.
- Reports Average Precision, recall, precision, calibration and captured fraud value.
- Selects its threshold on validation data, never on the final test set.
- Evaluates a limited-review-capacity scenario.
- Reports assumed business cost transparently.
- Serves reproducible predictions through a tested API.
- Detects schema failures and distribution drift.

No fixed performance target is declared before observing the temporal test distribution.

## Cost assumptions

Default illustrative scenario:

| Event | Assumed cost |
|---|---:|
| Manual review | EUR 5 |
| False positive | EUR 20 |
| Missed fraud fixed cost | EUR 500 |
| Potentially recoverable transaction amount | 80% |

These are scenario inputs, not real company costs or realised savings. Sensitivity analysis must show how the recommendation changes under conservative, medium and severe assumptions.

## Operational constraint

The secondary policy assumes that only the highest-risk 1% of transactions can be reviewed. The project reports precision and recall at that capacity in addition to the minimum-cost threshold.

## Non-goals

- Claiming production readiness for a real financial institution.
- Proving causal drivers of fraud.
- Identifying real people or accounts.
- Optimising accuracy.
- Using post-outcome information in the online model.

