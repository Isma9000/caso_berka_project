from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from caso_berka_model.config import PROJ_ROOT

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "Berka_Credit_Classification"
DEFAULT_REGISTERED_MODEL_NAME = "Berka_BuenCliente"


def resolve_tracking_uri(uri: str | None = None) -> str:
    """Resuelve la URI de tracking; rutas SQLite relativas se anclan a la raíz del repo."""
    resolved = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    prefix = "sqlite:///"
    if not resolved.startswith(prefix):
        return resolved

    db_path = resolved[len(prefix) :]
    path = Path(db_path)
    if not path.is_absolute():
        path = (PROJ_ROOT / path).resolve()
    return f"sqlite:///{path}"


def sanitize_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convierte hiperparámetros a tipos aceptados por MLflow."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        if value is None:
            sanitized[str(key)] = "None"
        elif isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def get_mlflow_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict((params or {}).get("mlflow") or {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("tracking_uri")
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "tracking_uri": resolve_tracking_uri(tracking_uri),
        "experiment_name": cfg.get("experiment_name", DEFAULT_EXPERIMENT_NAME),
        "registered_model_name": cfg.get(
            "registered_model_name", DEFAULT_REGISTERED_MODEL_NAME
        ),
        "decision_threshold": float(cfg.get("decision_threshold", 0.65)),
        "promote_to_production": bool(cfg.get("promote_to_production", True)),
        "run_evaluate": bool(cfg.get("run_evaluate", True)),
        "high_confidence": float(cfg.get("high_confidence", 0.85)),
    }
