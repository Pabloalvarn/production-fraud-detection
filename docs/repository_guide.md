# Repository guide

## Root files

| Path | Purpose |
|---|---|
| `README.md` | Recruiter-facing overview and execution instructions |
| `pyproject.toml` | Package metadata, dependencies and quality-tool configuration |
| `requirements.lock` | Compatibility dependency list; regenerate exact pins locally |
| `.env.example` | Non-secret runtime configuration template |
| `.gitignore` | Prevents committing data, models, credentials and temporary files |
| `Makefile` | Short commands for installation, training, tests and serving |
| `Dockerfile` | Reproducible API container |
| `docker-compose.yml` | Local API orchestration with a mounted model artifact |
| `LICENSE` | MIT licence for repository source code |

## Automation

| Path | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Lint, formatting and tests on pushes and pull requests |
| `.github/workflows/docker.yml` | Confirms that the container image builds |

## Configuration

| Path | Purpose |
|---|---|
| `configs/base.yaml` | Shared paths, split ratios, features and cost assumptions |
| `configs/model_logistic.yaml` | Interpretable Logistic Regression baseline |
| `configs/model_xgboost.yaml` | XGBoost candidate model |

## Data

| Path | Purpose |
|---|---|
| `data/README.md` | Dataset source, acquisition, schema and governance |
| `data/external/` | Unmodified PaySim CSV; ignored by Git |
| `data/interim/` | Temporary validated or sampled tables; ignored by Git |
| `data/processed/` | Derived model-ready tables; ignored by Git |

## Documentation

| Path | Purpose |
|---|---|
| `docs/problem_definition.md` | Decision, stakeholders, costs and success criteria |
| `docs/data_card.md` | Dataset lineage, intended use, leakage and limitations |
| `docs/model_card.md` | Governed model report template populated after training |
| `docs/architecture.md` | Training, inference and optional cloud architecture |
| `docs/monitoring_plan.md` | Data, drift, score, performance and policy monitoring |
| `docs/limitations.md` | Responsible-use and external-validity limitations |
| `docs/repository_guide.md` | This inventory |

## Notebooks

| Path | Purpose |
|---|---|
| `notebooks/01_data_validation.ipynb` | Inspect the data contract and quality report |
| `notebooks/02_exploratory_analysis.ipynb` | Study prevalence, amounts, types and time |
| `notebooks/03_feature_research.ipynb` | Research point-in-time-safe features |
| `notebooks/04_model_analysis.ipynb` | Interpret locked outputs without retuning on test data |

## Python package

| Module | Purpose |
|---|---|
| `config.py` | Load inherited YAML configurations and resolve paths |
| `data/download.py` | Kaggle or manual PaySim acquisition |
| `data/validation.py` | Hard schema checks and quality warnings |
| `data/split.py` | Whole-step chronological partitions |
| `features/build_features.py` | Online feature definitions and safe ratios |
| `features/transformers.py` | Scikit-learn-compatible feature builder |
| `models/train.py` | Complete training, calibration, threshold and test workflow |
| `models/calibrate.py` | Calibration on a later chronological partition |
| `models/predict.py` | Bundle loading and three-level decision policy |
| `models/registry.py` | MLflow experiment tracking |
| `evaluation/baselines.py` | Constant and rule-based baselines |
| `evaluation/metrics.py` | Rare-event, calibration, capacity and value metrics |
| `evaluation/business_cost.py` | Explicit scenario-based decision costs |
| `evaluation/threshold.py` | Cost and capacity threshold selection |
| `evaluation/plots.py` | PR, ROC and calibration figures |
| `evaluation/explain.py` | Locked-test permutation importance |
| `monitoring/drift.py` | PSI and KS drift report |
| `monitoring/performance.py` | Delayed-label performance report |
| `api/schemas.py` | Pydantic request and response contracts |
| `api/main.py` | FastAPI endpoints and model loading |

## Tests and outputs

| Path | Purpose |
|---|---|
| `tests/` | Unit, API and end-to-end smoke tests using synthetic fixtures |
| `artifacts/` | Generated model bundle; ignored by Git |
| `reports/metrics/` | Generated validation, baseline and model results |
| `reports/figures/` | Generated evaluation plots |
| `reports/executive_summary.md` | Business summary template completed with locked results |

## Files requiring user completion

The code and templates are complete, but the following cannot be truthfully pre-populated before running PaySim:

1. Numeric results in `docs/model_card.md`.
2. Numeric results and recommendation in `reports/executive_summary.md`.
3. Generated figures and metrics.
4. The final model artifact.
5. The GitHub username in the README badge.

These are intentionally left as `TBD` rather than fabricated.
