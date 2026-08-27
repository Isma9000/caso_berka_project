"""Tests de la API FastAPI con modelo mock (sin mlflow.db real)."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from caso_berka_model.api import main as api_main
from caso_berka_model.api.schemas import FEATURE_COLUMNS


class _MockPyFuncModel:
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
def client_with_model(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: _MockPyFuncModel())
    monkeypatch.setattr(
        api_main,
        "get_model_metadata",
        lambda: {"version": "1", "run_id": "test-run"},
    )
    with TestClient(api_main.app) as client:
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda: None)
    monkeypatch.setattr(
        api_main,
        "get_model_metadata",
        lambda: {"version": "Desconocida", "run_id": "Desconocido"},
    )
    with TestClient(api_main.app) as client:
        yield client


def test_root_online(client_with_model):
    response = client_with_model.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "Online"
    assert body["production_version"] == "1"
    assert body["run_id"] == "test-run"


def test_health_ok(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_unavailable_without_model(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 503


def test_predict_success(client_with_model, sample_payload):
    response = client_with_model.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 1
    assert body["results"][0]["prediction"] == 1
    assert body["results"][0]["diagnosis"] == "Buen cliente"
    assert body["results"][0]["probability"] == 0.9
    assert body["results"][0]["high_confidence_flag"] == 1
    assert body["model_metadata"]["version"] == "1"
    assert set(FEATURE_COLUMNS) == set(sample_payload["data"][0].keys())


def test_predict_without_model_returns_500(client_without_model, sample_payload):
    response = client_without_model.post("/predict", json=sample_payload)
    assert response.status_code == 500
