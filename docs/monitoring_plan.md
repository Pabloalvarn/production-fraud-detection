# Monitoring plan

## Objectives

Monitoring must determine whether inputs remain valid, the population has changed, operational capacity is respected and predictive performance remains acceptable once confirmed labels arrive.

## Monitoring layers

### 1. Service health

| Signal | Frequency | Example alert |
|---|---|---|
| API availability | Continuous | Error rate above 1% |
| Latency | Continuous | p95 above agreed limit |
| Model loaded | On startup/health check | `model_loaded=false` |
| Invalid requests | Daily | Sudden increase from baseline |

### 2. Data quality

Monitor on every batch:

- Required columns.
- Data types.
- Missing values.
- Negative amounts.
- Unknown transaction types.
- Infinite derived features.
- Changes in row volume.

Hard schema violations reject the batch. Distribution changes generate warnings rather than silently changing the model.

### 3. Feature and score drift

| Signal | Suggested threshold | Action |
|---|---:|---|
| PSI below 0.10 | Low | Continue monitoring |
| PSI 0.10–0.20 | Moderate | Investigate |
| PSI at least 0.20 | High | Escalate and assess retraining |
| Unknown categories | Above 0 | Investigate data contract |
| Review rate | Above capacity | Reassess threshold/policy |

PSI thresholds are heuristics and must be validated for the specific production system. The repository also reports the Kolmogorov–Smirnov statistic and p-value for continuous features; statistical significance alone is not an operational decision.

### 4. Delayed-label performance

When confirmed fraud outcomes arrive, calculate:

- Average Precision.
- Precision and recall at the locked threshold.
- Precision and recall at review capacity.
- Brier score and calibration.
- Fraud value captured.
- Expected cost under the documented scenario.
- Segment performance by transaction type, amount and time.

### 5. Policy monitoring

Track:

- Percentage approved.
- Percentage reviewed.
- Percentage requiring enhanced verification.
- False-positive burden.
- Queue size and review turnaround time.
- Changes in thresholds and cost assumptions.

## Retraining trigger

Retraining is considered—not automatically executed—when one or more of these persist:

- Material feature drift.
- Average Precision deterioration relative to baseline.
- Recall below the business floor.
- Poor calibration.
- New transaction categories.
- Sustained review-capacity breach.

Every retrained candidate must repeat chronological validation and be compared with the incumbent model before promotion.

## Model promotion

1. Train candidate on an approved window.
2. Calibrate using later observations.
3. Select policy threshold on a separate validation period.
4. Compare against incumbent on untouched data.
5. Document metrics and limitations.
6. Approve manually.
7. Deploy with versioned rollback.

## Portfolio limitation

This plan demonstrates monitoring design. It does not claim that the synthetic dataset provides realistic alert thresholds or retraining frequency.

