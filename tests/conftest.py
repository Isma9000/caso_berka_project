"""Fixtures compartidas para la suite de tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from caso_berka_model.api import main as api_main
from caso_berka_model.api.schemas import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


class MockPyFuncModel:
    """Simula el contrato del EnterpriseDecisionWrapper / PyFunc Production."""

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        n = len(model_input)
        return pd.DataFrame(
            {
                "probability": [0.9] * n,
                "prediction": [1] * n,
                "high_confidence_flag": [1] * n,
            }
        )


@pytest.fixture
def sample_payload():
    return {
        "data": [
            {
                "birth_number": 591001.0,
                "date": 931008.0,
                "cantidad_ingresos": 0.97,
                "total_egresos": -0.18,
                "cantidad_egresos": 1.94,
                "tiene_prestamo": 0.0,
                "monto_prestamo": -0.33,
                "tiene_tarjeta": 0.0,
            }
        ]
    }


@pytest.fixture
def api_client_with_model(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: MockPyFuncModel())
    monkeypatch.setattr(
        api_main,
        "get_model_metadata",
        lambda: {"version": "1", "run_id": "test-run"},
    )
    with TestClient(api_main.app) as client:
        yield client


@pytest.fixture
def api_client_without_model(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: None)
    monkeypatch.setattr(
        api_main,
        "get_model_metadata",
        lambda: {"version": "Desconocida", "run_id": "Desconocido"},
    )
    with TestClient(api_main.app) as client:
        yield client


@pytest.fixture
def synthetic_tabla_minable() -> pd.DataFrame:
    """Tabla mínima compatible con ModelTrainer.separar_variables."""
    rows = []
    for i in range(60):
        label = i % 2
        rows.append(
            {
                "client_id": i,
                "disp_id": i,
                "account_id": i,
                "birth_number": 591001.0 + i,
                "date": 931008.0 + i,
                "cantidad_ingresos": 0.5 + label,
                "total_egresos": -0.1,
                "cantidad_egresos": 1.0,
                "tiene_prestamo": float(label),
                "monto_prestamo": -0.2,
                "tiene_tarjeta": float(1 - label),
                "total_ingresos": 100.0 + label,
                "total_transacciones": 20.0 + label,
                "saldo_promedio": 10.0,
                "moroso": 0,
                "buen_cliente": label,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_params_yaml(tmp_path: Path) -> Path:
    """Copia params.yaml al directorio temporal con MLflow desactivado."""
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    params["mlflow"]["enabled"] = False
    params["train"]["n_estimators"] = 10
    params["train"]["knn_k_max"] = 5
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return params_path


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def feature_columns():
    return FEATURE_COLUMNS
