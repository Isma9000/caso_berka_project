"""API FastAPI para desplegar Berka_BuenCliente desde el Model Registry."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
import pandas as pd

from caso_berka_model.api.model_loader import (
    MODEL_NAME,
    get_model_metadata,
    load_model,
)
from caso_berka_model.api.schemas import (
    FEATURE_COLUMNS,
    PredictionRequest,
    PredictionResponse,
)

model: Any = None
model_version_info: dict[str, str] = {
    "version": "Desconocida",
    "run_id": "Desconocido",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, model_version_info
    model = load_model()
    model_version_info = get_model_metadata()
    if model is not None:
        print(
            f"[FastAPI] Modelo v{model_version_info['version']} "
            "cargado exitosamente en producción!"
        )
    else:
        print("[FastAPI ERROR] El modelo inició en None.")
    yield


app = FastAPI(
    title="API de Clasificación Berka — Buen Cliente",
    description=(
        "Inferencia del modelo Berka_BuenCliente (Production) con arquitectura modular MLOps."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "model_name": MODEL_NAME,
        "production_version": model_version_info["version"],
        "run_id": model_version_info["run_id"],
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado en memoria.",
        )
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "production_version": model_version_info["version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=("El modelo no está cargado en memoria o no se encontró en el Model Registry."),
        )

    try:
        input_data = pd.DataFrame(
            [item.model_dump() for item in payload.data],
            columns=FEATURE_COLUMNS,
        )
        predictions = model.predict(input_data)

        results = []
        for idx, row in enumerate(predictions.to_dict(orient="records")):
            pred = int(row["prediction"])
            prob = float(row["probability"])
            high_conf = int(row["high_confidence_flag"])
            diagnosis = "Buen cliente" if pred == 1 else "Mal cliente"
            results.append(
                {
                    "index": idx,
                    "prediction": pred,
                    "diagnosis": diagnosis,
                    "probability": round(prob, 4),
                    "high_confidence_flag": high_conf,
                    "confidence_score": round(prob * 100, 2),
                }
            )

        return {
            "model_metadata": {
                "name": MODEL_NAME,
                "version": model_version_info["version"],
                "run_id": model_version_info["run_id"],
            },
            "total_predictions": len(results),
            "results": results,
            "message": "Inferencia completada con éxito.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error durante la inferencia: {exc}",
        ) from exc
