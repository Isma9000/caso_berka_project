# Assets para la documentación y la presentación

Coloca aquí capturas de pantalla (PNG/JPG) referenciadas desde [06-mlflow.md](../06-mlflow.md), [08-fastapi.md](../08-fastapi.md) y [slides.md](../slides.md).

## Nombres sugeridos

| Archivo | Contenido |
|---------|-----------|
| `mlflow-experiment.png` | Lista / detalle del experimento `Berka_Credit_Classification` |
| `mlflow-nested-runs.png` | Runs anidados de los 4 modelos |
| `mlflow-registry-production.png` | Modelo `Berka_BuenCliente` en Production |
| `swagger-docs.png` | Swagger UI en `/docs` |
| `health-endpoint.png` | Respuesta de `GET /health` |
| `pytest-output.png` | Terminal con tests en verde |
| `dvc-metrics.png` | Salida de `dvc metrics show` |

## Cómo enlazar en Markdown (MkDocs)

```markdown
![MLflow Registry](assets/mlflow-registry-production.png)
```

## Figuras del entrenamiento

Las gráficas generadas por el pipeline viven en `reports/figures/` en la raíz del repo (ignoradas por Git). Regenerar con:

```bash
make train
```

No es obligatorio copiarlas aquí; basta con adjuntarlas a las diapositivas o regenerarlas antes de la demo.
