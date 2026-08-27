"""Esquemas Pydantic para la API de inferencia Berka."""

from __future__ import annotations

from pydantic import BaseModel, Field

FEATURE_COLUMNS = [
    "birth_number",
    "date",
    "cantidad_ingresos",
    "total_egresos",
    "cantidad_egresos",
    "tiene_prestamo",
    "monto_prestamo",
    "tiene_tarjeta",
]


class BerkaFeatures(BaseModel):
    """Features ya preprocesadas (mismo contrato que el PyFunc en Production)."""

    birth_number: float
    date: float
    cantidad_ingresos: float
    total_egresos: float
    cantidad_egresos: float
    tiene_prestamo: float
    monto_prestamo: float
    tiene_tarjeta: float


class PredictionRequest(BaseModel):
    data: list[BerkaFeatures] = Field(..., min_length=1)


class ModelMetadata(BaseModel):
    name: str
    version: str
    run_id: str


class PredictionResult(BaseModel):
    index: int
    prediction: int
    diagnosis: str
    probability: float
    high_confidence_flag: int
    confidence_score: float | None = None


class PredictionResponse(BaseModel):
    model_metadata: ModelMetadata
    total_predictions: int
    results: list[PredictionResult]
    message: str
