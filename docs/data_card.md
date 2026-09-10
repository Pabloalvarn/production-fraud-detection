# Data card — PaySim

## Summary

PaySim is a synthetic dataset designed to resemble mobile-money transactions. Its generator was informed by aggregated patterns from a private transactional system and adds simulated malicious behaviour. The commonly distributed sample contains millions of records and a highly imbalanced binary fraud label.

## Intended use in this repository

- Demonstrate rare-event classification.
- Compare baselines and tree-based models.
- Apply temporal holdout validation.
- Study probability calibration and threshold policies.
- Demonstrate serving, testing and monitoring patterns.

## Inappropriate uses

- Real credit, payment or fraud decisions.
- Claims about a population, country or protected group.
- Estimates of real fraud prevalence.
- Claims of real financial savings.
- Assessment of individual customers.

## Unit of observation

One simulated transaction.

## Target

`isFraud`, where 1 denotes an injected simulated fraudulent transaction.

## Point-in-time availability

| Feature | Assumption | Decision |
|---|---|---|
| `step` | Available | Use |
| `type` | Available | Use |
| `amount` | Available | Use |
| `oldbalanceOrg` | Available | Use |
| `oldbalanceDest` | Available | Use |
| `newbalanceOrig` | Post-transaction or ambiguous | Exclude online |
| `newbalanceDest` | Post-transaction or ambiguous | Exclude online |
| `isFlaggedFraud` | Output of another policy | Exclude |
| `nameOrig`, `nameDest` | High-cardinality IDs | Exclude raw |

## Splitting policy

Rows are sorted by `step`. Entire steps are allocated to train, validation and test partitions so the same step does not cross boundaries. The validation partition is divided chronologically: its earlier half calibrates probabilities and its later half selects the threshold. The test partition remains untouched until the final evaluation.

## Known limitations

- Synthetic behaviour may be easier to distinguish than real fraud.
- Only a small number of transaction attributes is provided.
- `step` is a simulated time index rather than a real timestamp.
- Identity fields do not include genuine customer history or demographics.
- Label-generation rules can introduce artificial patterns.
- Real labels are often delayed, incomplete and affected by investigation policy.

## Ethical and privacy considerations

No real personal information is expected in PaySim. Raw IDs are nevertheless treated as potentially sensitive identifiers and are not exposed by the API or committed to Git. A real deployment would require privacy review, access controls, retention limits, fairness analysis and human appeal mechanisms.

## Validation checks

The pipeline checks required columns, binary target values, transaction categories, negative amounts, nulls, duplicates and temporal ordering. Validation reports are written to `reports/metrics/` and are not used as a substitute for source-data governance.

