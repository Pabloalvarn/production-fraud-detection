# Data access and governance

This repository uses **PaySim**, a synthetic mobile-money transaction dataset created to support fraud-detection research. PaySim was generated from aggregated patterns observed in one month of private mobile-money logs and injects fraudulent behaviour into simulated transactions. The project does not distribute the dataset.

## Source

- Kaggle dataset: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- Expected file: `PS_20174392719_1491204439457_log.csv`
- Background: E. A. Lopez-Rojas, A. Elmir and S. Axelsson, *PaySim: A Financial Mobile Money Simulator for Fraud Detection*.

Review the dataset page and its current terms before downloading or redistributing any files.

## Acquisition

### Option A — manual download

1. Download and unzip the dataset from Kaggle.
2. Run:

```bash
python -m fraud_detection.data.download \
  --source-path "/path/to/PS_20174392719_1491204439457_log.csv"
```

### Option B — Kaggle CLI

Configure the Kaggle CLI and run:

```bash
python -m fraud_detection.data.download --kaggle
```

## Directory policy

- `external/`: untouched source CSV.
- `interim/`: temporary validated or sampled data.
- `processed/`: model-ready derived tables.

All three directories are ignored by Git except for `.gitkeep`. Never commit raw data, generated predictions containing identifiers, access tokens or Kaggle credentials.

## Expected schema

| Column | Type | Meaning | Online model |
|---|---|---|---|
| `step` | integer | Simulated time step | Included |
| `type` | category | Transaction type | Included |
| `amount` | float | Transaction amount | Included |
| `nameOrig` | string | Origin account identifier | Excluded; potential historical aggregates only |
| `oldbalanceOrg` | float | Origin balance before transaction | Included |
| `newbalanceOrig` | float | Origin balance after transaction | Excluded from pre-authorisation model |
| `nameDest` | string | Destination identifier | Excluded; potential historical aggregates only |
| `oldbalanceDest` | float | Destination balance before transaction | Included |
| `newbalanceDest` | float | Destination balance after transaction | Excluded from pre-authorisation model |
| `isFraud` | binary | Simulated fraud label | Target only |
| `isFlaggedFraud` | binary | Existing simulated rule flag | Excluded to prevent policy leakage |

