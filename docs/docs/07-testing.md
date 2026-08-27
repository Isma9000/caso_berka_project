# 7. Estrategia de testing (Pytest)

## Cómo ejecutar

```bash
make test
# o
pytest tests
```

Dependencias: `pytest`, `httpx` (cliente ASGI de FastAPI), `scikit-learn`, `mlflow`.

## Organización por contrato

| Archivo | Alcance | Tipo |
|---------|---------|------|
| `tests/test_data.py` | Objetivo `buen_cliente` y existencia de clases del pipeline | Unitario |
| `tests/test_api.py` | Endpoints FastAPI con modelo mock | Unitario / contrato API |
| `tests/test_mlflow.py` | Linaje, PyFunc, tracking SQLite temporal, registry | Unitario + integración aislada |

## Detalle de pruebas

### Datos (`test_data.py`)

- `test_feature_engineer_objetivo`: DataFrame sintético → columna `buen_cliente` ∈ {0, 1}.
- `test_clases_pipeline_existen`: importa `DataProcessor`, `FeatureEngineer`, `ModelTrainer`, `Evaluator`, `Predictor`.

### API (`test_api.py`)

Usa `TestClient` y un `_MockPyFuncModel` que imita el contrato Production (`probability`, `prediction`, `high_confidence_flag`).

| Test | Expectativa |
|------|-------------|
| `GET /` | 200, status Online, versión y `run_id` |
| `GET /health` con modelo | 200, `healthy` |
| `GET /health` sin modelo | **503** |
| `POST /predict` con modelo | 200, diagnóstico, probabilidad, metadata |
| `POST /predict` sin modelo | **500** |
| Payload | Las **8** features de `FEATURE_COLUMNS` |

### MLflow (`test_mlflow.py`)

| Test | Expectativa |
|------|-------------|
| `test_lineage_metadata` | Tags `git_commit` y `dvc_status` presentes |
| `test_pyfunc_wrapper_predict` | Columnas del wrapper de decisión |
| `test_mlflow_logs_metrics_and_registry` | DB SQLite temporal, métricas en runs, promoción Production, `predict` |

!!! note "Aislamiento"
    Los tests de MLflow usan `tmp_path` y **no** dependen del `mlflow.db` real del desarrollador.

## Criterio de calidad pre-merge

Antes de integrar cambios se recomienda:

```bash
pytest tests
ruff check
dvc repro          # si cambian datos/código del pipeline
curl .../health    # si cambia serving
```

## Siguiente lectura

- [8. FastAPI](08-fastapi.md)
- [Diapositivas 8–9](slides.md#diapositivas-8-9)
