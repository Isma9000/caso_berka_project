# 8. Productivización con FastAPI

## Aplicación

- Módulo: `caso_berka_model/api/main.py`
- Carga del modelo: ciclo de vida `lifespan` → `load_model()` / `get_model_metadata()`
- Local: Model Registry `Berka_BuenCliente@Production`
- Docker: ruta fija `models/docker_production` (`ENVIRONMENT=docker`)

```bash
make api-serve
# http://127.0.0.1:8000/docs  → Swagger UI
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Estado Online, nombre, versión productiva, `run_id` |
| `GET` | `/health` | `healthy` o **HTTP 503** si no hay modelo |
| `POST` | `/predict` | Inferencia batch (mínimo 1 fila en `data`) |

## Esquemas Pydantic (`schemas.py`)

### Entrada — `BerkaFeatures` / `PredictionRequest`

Ocho features **ya preprocesadas** (mismo contrato que el PyFunc):

```text
birth_number, date, cantidad_ingresos, total_egresos,
cantidad_egresos, tiene_prestamo, monto_prestamo, tiene_tarjeta
```

`PredictionRequest.data`: `list[BerkaFeatures]` con `min_length=1`.

### Salida — `PredictionResponse`

- `model_metadata`: `name`, `version`, `run_id`
- `total_predictions`
- `results[]`: `index`, `prediction`, `diagnosis` ("Buen cliente" / "Mal cliente"), `probability`, `high_confidence_flag`, `confidence_score`
- `message`

## Swagger UI

Al levantar Uvicorn, la documentación interactiva OpenAPI está en:

**http://127.0.0.1:8000/docs**

Útil para probar `POST /predict` sin `curl`. Captura recomendada en [`assets/`](assets/README.md).

## Ejemplo de inferencia

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [{
      "birth_number": 591001.0,
      "date": 931008.0,
      "cantidad_ingresos": 0.97,
      "total_egresos": -0.18,
      "cantidad_egresos": 1.94,
      "tiene_prestamo": 0.0,
      "monto_prestamo": -0.33,
      "tiene_tarjeta": 0.0
    }]
  }'
```

## Limitación conocida

La API **no** transforma tablas Berka crudas: exige features ya alineadas con el entrenamiento. Una mejora futura es persistir el pipeline de preprocesamiento y exponer un endpoint de datos crudos (ver INFORME, propuestas de mejora).

## Siguiente lectura

- [9. Docker](09-docker.md)
- [Diapositivas 8–9](slides.md#diapositivas-8-9)
