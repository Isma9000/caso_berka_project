"""Tests de la API FastAPI con modelo mock (sin mlflow.db real)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_root_online(api_client_with_model):
    response = api_client_with_model.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "Online"
    assert body["production_version"] == "1"
    assert body["run_id"] == "test-run"


def test_health_ok(api_client_with_model):
    response = api_client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_unavailable_without_model(api_client_without_model):
    response = api_client_without_model.get("/health")
    assert response.status_code == 503


def test_predict_success(api_client_with_model, sample_payload, feature_columns):
    response = api_client_with_model.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 1
    assert body["results"][0]["prediction"] == 1
    assert body["results"][0]["diagnosis"] == "Buen cliente"
    assert body["results"][0]["probability"] == 0.9
    assert body["results"][0]["high_confidence_flag"] == 1
    assert body["model_metadata"]["version"] == "1"
    assert set(feature_columns) == set(sample_payload["data"][0].keys())


def test_predict_without_model_returns_500(api_client_without_model, sample_payload):
    response = api_client_without_model.post("/predict", json=sample_payload)
    assert response.status_code == 500


def test_predict_invalid_payload_returns_422(api_client_with_model):
    response = api_client_with_model.post(
        "/predict",
        json={"data": [{"birth_number": 1.0}]},
    )
    assert response.status_code == 422
