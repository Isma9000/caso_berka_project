"""Carga del modelo Production desde el Model Registry MLflow."""

from __future__ import annotations

from typing import Any

import yaml

from caso_berka_model.config import PARAMS_PATH
from caso_berka_model.mlflow_engine.registry import MLflowGovernanceManager
from caso_berka_model.mlflow_engine.settings import (
    DEFAULT_REGISTERED_MODEL_NAME,
    get_mlflow_config,
)

_UNKNOWN = {"version": "Desconocida", "run_id": "Desconocido"}


def _load_params() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        return {}
    with PARAMS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _mlflow_cfg() -> dict[str, Any]:
    return get_mlflow_config(_load_params())


MODEL_NAME: str = _mlflow_cfg()["registered_model_name"] or DEFAULT_REGISTERED_MODEL_NAME


def load_model():
    """Carga el PyFunc registrado en Production. Devuelve None si falla."""
    cfg = _mlflow_cfg()
    name = cfg["registered_model_name"]
    try:
        return MLflowGovernanceManager(cfg["tracking_uri"]).load_latest_production_model(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[API] Error cargando modelo '{name}': {exc}")
        return None


def get_model_metadata() -> dict[str, str]:
    """Version y run_id del modelo en Production."""
    cfg = _mlflow_cfg()
    name = cfg["registered_model_name"]
    try:
        return MLflowGovernanceManager(cfg["tracking_uri"]).get_production_metadata(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[API] Error obteniendo metadata de '{name}': {exc}")
        return dict(_UNKNOWN)
