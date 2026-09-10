"""Validated API contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Transaction(BaseModel):
    step: int = Field(ge=1, description="PaySim time step.")
    type: TransactionType
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)

    @field_validator("amount", "oldbalanceOrg", "oldbalanceDest")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("value must be finite")
        return value


class Prediction(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "manual_review", "enhanced_verification"]
    model_version: str


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: str | None = None
