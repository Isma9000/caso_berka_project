# 7. Estrategia de testing (Pytest)

## Cómo ejecutar

```bash
make test-unit          # rápido: solo unit/
make test-integration   # integración aislada
make test               # todo excepto slow (e2e Docker)
make ci-local           # lint + unit + integration (pre-merge)
```

Equivalente directo:

```bash
pytest tests/unit -m unit
pytest tests/integration -m integration
pytest tests -m "not slow"
```

Dependencias: `pytest`, `httpx`, `scikit-learn`, `mlflow`. Configuración en [`pyproject.toml`](../pyproject.toml) (`markers`, `testpaths`).

## Organización por capas

```
tests/
  conftest.py           # fixtures compartidas
  unit/                 # @pytest.mark.unit
  integration/          # @pytest.mark.integration
  e2e/                  # @pytest.mark.slow (Docker)
```

| Capa | Archivos | Tipo | Qué valida |
|------|----------|------|------------|
| `unit/` | `test_features.py`, `test_train.py`, `test_evaluator.py`, `test_predictor.py`, `test_schemas.py`, `test_api.py`, `test_mlflow_pyfunc.py` | Unitario | Lógica aislada, contrato API con mock, PyFunc y linaje |
| `integration/` | `test_train_pipeline.py`, `test_mlflow_registry.py`, `test_dvc_contract.py` | Integración aislada | Entrenamiento en `tmp_path`, registry MLflow temporal, contrato DVC |
| `e2e/` | `test_docker_health.py` | E2E lento | Health del contenedor (skip si no hay Docker/imagen) |

## Detalle de pruebas clave

### Datos y entrenamiento (`unit/`)

- `test_feature_engineer_objetivo`: columna `buen_cliente` ∈ {0, 1}
- `test_prepare_data_uses_prepare_split_ratio`: regresión del esquema `prepare.split_ratio` (no `split`)
- `test_save_dvc_metrics_writes_eval_json`: `eval.json` y `plots.csv` en directorio temporal
- `test_decile_table_returns_ten_deciles`: tabla de deciles con 10 filas

### API (`unit/test_api.py`)

Usa `TestClient` y `MockPyFuncModel` (sin `mlflow.db` real).

| Test | Expectativa |
|------|-------------|
| `GET /` | 200, status Online |
| `GET /health` con modelo | 200, `healthy` |
| `GET /health` sin modelo | **503** |
| `POST /predict` con modelo | 200, diagnóstico y metadata |
| `POST /predict` sin modelo | **500** |
| `POST /predict` payload incompleto | **422** |

### MLflow (`unit/` + `integration/`)

| Test | Capa | Expectativa |
|------|------|-------------|
| `test_lineage_metadata` | unit | Tags `git_commit`, `dvc_status` |
| `test_pyfunc_wrapper_predict` | unit | Columnas del wrapper |
| `test_pyfunc_wrapper_uses_custom_threshold` | unit | Umbral de decisión personalizado |
| `test_mlflow_logs_metrics_and_registry` | integration | SQLite temporal, métricas, Production, `predict` |

### DVC (`integration/test_dvc_contract.py`)

- `dvc.lock` sin marcadores de merge
- `dvc.yaml`, `params.yaml`, `dvc.lock` parseables como YAML
- `dvc status` (marcado `slow`, opcional en local)

!!! note "Aislamiento"
    Los tests de integración usan `tmp_path` y datos sintéticos. **No** dependen del remote DVC ni del `mlflow.db` del desarrollador.

## Criterio de calidad pre-merge

```bash
make ci-local         # desarrollo diario
dvc repro             # si cambian datos/código del pipeline
curl .../health       # si cambia serving o Docker
```

## Siguiente lectura

- [8. FastAPI](08-fastapi.md)
- [Diapositivas 8–9](slides.md#diapositivas-8-9)
