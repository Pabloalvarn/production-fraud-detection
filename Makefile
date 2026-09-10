.PHONY: install download validate train-logistic train-xgboost test lint format api docker clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

download:
	python -m fraud_detection.data.download --kaggle

validate:
	python -m fraud_detection.data.validation --config configs/base.yaml

train-logistic:
	python -m fraud_detection.models.train --config configs/model_logistic.yaml

train-xgboost:
	python -m fraud_detection.models.train --config configs/model_xgboost.yaml

test:
	pytest --cov=fraud_detection --cov-report=term-missing

lint:
	ruff check .
	mypy src/fraud_detection

format:
	ruff format .
	ruff check --fix .

api:
	uvicorn fraud_detection.api.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['artifacts','mlruns','.pytest_cache','.ruff_cache','htmlcov']]"

