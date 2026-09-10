"""HTTP inference API for the fraud model."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from fraud_detection.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    Prediction,
    Transaction,
)
from fraud_detection.config import project_path
from fraud_detection.models.predict import load_bundle, predict_frame

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


def configured_model_path() -> Path:
    return project_path(os.getenv("MODEL_PATH", "artifacts/model_bundle.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    path = configured_model_path()
    try:
        app.state.bundle = load_bundle(path)
        LOGGER.info("Loaded model bundle %s", path)
    except (FileNotFoundError, ValueError) as error:
        app.state.bundle = None
        LOGGER.warning("API started without a model: %s", error)
    yield


app = FastAPI(
    title="Production Fraud Detection API",
    version="0.1.0",
    description=(
        "Portfolio demonstration only. The model is trained on synthetic PaySim data and "
        "must not be used to make real financial decisions."
    ),
    lifespan=lifespan,
)


def get_bundle(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; train it first.")
    return bundle


def transaction_frame(transactions: list[Transaction]) -> pd.DataFrame:
    return pd.DataFrame([transaction.model_dump() for transaction in transactions])


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    bundle = getattr(request.app.state, "bundle", None)
    return HealthResponse(
        status="healthy" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.get("model_version") if bundle else None,
    )


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    bundle = get_bundle(request)

    training_end_step = bundle.get("training_end_step")
    validation_end_step = bundle.get("validation_end_step")
    high_risk_threshold = bundle.get("high_risk_threshold")

    return {
        "model_version": str(bundle["model_version"]),
        "created_at_utc": bundle.get("created_at_utc"),
        "threshold": float(bundle["threshold"]),
        "high_risk_threshold": (
            float(high_risk_threshold) if high_risk_threshold is not None else None
        ),
        "raw_features": list(bundle["raw_features"]),
        "training_end_step": (int(training_end_step) if training_end_step is not None else None),
        "validation_end_step": (
            int(validation_end_step) if validation_end_step is not None else None
        ),
        "limitations": ("Synthetic-data portfolio model; not approved for real decisions."),
    }


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request) -> Prediction:
    bundle = get_bundle(request)
    result = predict_frame(bundle, transaction_frame([transaction])).iloc[0].to_dict()
    LOGGER.info("prediction decision=%s risk=%s", result["decision"], result["risk_level"])
    return Prediction(**result)


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    bundle = get_bundle(request)
    results = predict_frame(bundle, transaction_frame(payload.transactions))
    return BatchResponse(predictions=[Prediction(**row) for row in results.to_dict("records")])
