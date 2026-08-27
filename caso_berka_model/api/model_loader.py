"""Carga del modelo Production: Registry (local) o ruta fija (Docker)."""

from __future__ import annotations

import os
from typing import Any

import mlflow.pyfunc
import yaml

from caso_berka_model.config import PARAMS_PATH
from caso_berka_model.mlflow_engine.registry import MLflowGovernanceManager
from caso_berka_model.mlflow_engine.settings import (
    DEFAULT_REGISTERED_MODEL_NAME,
    get_mlflow_config,
)

_UNKNOWN = {"version": "Desconocida", "run_id": "Desconocido"}

IS_DOCKER = os.environ.get("ENVIRONMENT") == "docker"
DOCKER_MODEL_PATH = os.environ.get("DOCKER_MODEL_PATH", "/app/models/docker_production")
# Defaults alineados con Berka_BuenCliente Production al momento de documentar Docker
DOCKER_MODEL_VERSION = os.environ.get("DOCKER_MODEL_VERSION", "5")
DOCKER_RUN_ID = os.environ.get("DOCKER_RUN_ID", "0629ea3b63574208ab318254e38ff748")


def _load_params() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        return {}
    with PARAMS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _mlflow_cfg() -> dict[str, Any]:
    return get_mlflow_config(_load_params())


MODEL_NAME: str = _mlflow_cfg()["registered_model_name"] or DEFAULT_REGISTERED_MODEL_NAME


def load_model():
    """Carga el PyFunc según el entorno (Docker: ruta fija; local: Registry)."""
    try:
        print(f"[ModelLoader] Entorno detectado: {'Docker' if IS_DOCKER else 'Local'}")

        if IS_DOCKER:
            print(f"[ModelLoader] Cargando PyFunc en Docker desde: {DOCKER_MODEL_PATH}")
            return mlflow.pyfunc.load_model(DOCKER_MODEL_PATH)

        cfg = _mlflow_cfg()
        name = cfg["registered_model_name"]
        return MLflowGovernanceManager(cfg["tracking_uri"]).load_latest_production_model(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[ModelLoader ERROR] No se pudo cargar el modelo: {exc}")
        return None


def get_model_metadata() -> dict[str, str]:
    """Version y run_id del modelo en Production."""
    if IS_DOCKER:
        return {
            "version": DOCKER_MODEL_VERSION,
            "run_id": DOCKER_RUN_ID,
        }

    cfg = _mlflow_cfg()
    name = cfg["registered_model_name"]
    try:
        return MLflowGovernanceManager(cfg["tracking_uri"]).get_production_metadata(name)
    except Exception as exc:  # noqa: BLE001
        print(f"[API] Error obteniendo metadata de '{name}': {exc}")
        return dict(_UNKNOWN)
