# Architecture

## Training workflow

```mermaid
flowchart TD
    A[PaySim CSV] --> B[Schema validation]
    B --> C[Chronological split]
    C --> D[Leakage-safe features]
    D --> E[Model training]
    E --> F[Probability calibration]
    F --> G[Threshold selection]
    G --> H[Locked test evaluation]
    H --> I[Model bundle and reports]
```

## Validation partitions

```mermaid
flowchart LR
    A[Train 70%] --> B[Calibration 7.5%]
    B --> C[Threshold 7.5%]
    C --> D[Test 15%]
```

All percentages are chronological approximations based on complete `step` values. No future partition is used to fit an earlier stage.

## Inference workflow

```mermaid
flowchart TD
    A[Transaction JSON] --> B[Pydantic validation]
    B --> C[Feature pipeline]
    C --> D[Calibrated classifier]
    D --> E[Risk probability]
    E --> F[Decision thresholds]
    F --> G[Approve / review / verify]
```

## Component responsibilities

| Component | Responsibility |
|---|---|
| Data validation | Reject incompatible datasets and report warnings |
| Temporal split | Prevent future observations leaking into training |
| Feature builder | Create pre-authorisation features |
| Preprocessor | Impute, scale and encode consistently |
| Estimator | Rank transactions by fraud risk |
| Calibrator | Improve probability interpretation |
| Threshold policy | Translate probabilities into operational actions |
| FastAPI | Validate and serve requests |
| Monitoring | Detect data, score and performance deterioration |

## Artifact contract

Training serialises one `joblib` bundle containing:

- Calibrated pipeline.
- Medium-risk threshold.
- High-risk threshold.
- Model version.
- Required raw features.
- Training and validation boundaries.
- Cost assumptions.
- Final test metrics.

The bundle is generated locally and excluded from Git because serialized model objects are environment-dependent and may be large.

## Future AWS deployment

```mermaid
flowchart TD
    A[Client] --> B[API Gateway]
    B --> C[App Runner / ECS]
    C --> D[Fraud model]
    C --> E[CloudWatch logs]
    E --> F[S3 monitoring data]
    F --> G[Scheduled drift report]
```

Cloud deployment is optional. A real architecture would additionally require authentication, encryption, secrets management, network isolation, audit trails, rate limiting and rollback procedures.

