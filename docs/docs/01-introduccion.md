# 1. Introducción y objetivos

## Planteamiento del problema de negocio

Las entidades financieras procesan grandes volúmenes de información de clientes, cuentas, transacciones, préstamos y tarjetas. Necesitan **priorizar clientes** con características compatibles con un buen comportamiento crediticio, de forma **consistente, trazable y reproducible**.

Este proyecto desarrolla un clasificador binario sobre el conjunto **Banco Berka** (PKDD'99):

| Clase | Significado operativo |
|-------|------------------------|
| `buen_cliente = 1` | Perfil compatible con buen comportamiento crediticio (según regla de negocio) |
| `buen_cliente = 0` | Perfil no compatible con esa regla |

El resultado es una **herramienta de apoyo a la decisión**, no un sustituto de la evaluación financiera, normativa o humana.

!!! danger "No es un detector de fraude"
    La variable objetivo **no** representa fraude observado. Es una etiqueta de comportamiento crediticio construida a partir de ingresos, actividad transaccional, saldo y morosidad. Interpretar las métricas como evidencia de detección de fraude sería incorrecto.

## Problema de ingeniería

Integrar tablas relacionadas, resumir transacciones, tratar faltantes, codificar categóricas, evitar que identificadores dominen el aprendizaje y **comparar modelos** con una métrica adecuada, dentro de un ciclo MLOps reproducible.

### Objetivos de ingeniería

| Objetivo | Cómo se aborda en el proyecto |
|----------|-------------------------------|
| Versionar datos y modelos pesados | DVC (`dvc.yaml`, remote local); Git guarda código, config y punteros |
| Experimentos y registro de modelos | MLflow Tracking + Model Registry (`Berka_BuenCliente`) |
| Código mantenible | Clases con responsabilidades separadas y estilo PEP8 (Ruff) |
| Calidad | Pytest (datos, API, linaje MLflow) |
| Productivización | API REST FastAPI + Pydantic |
| Despliegue aislado | Docker / Docker Compose (puerto 8000) |

## Métrica principal: F1

Se eligió **F1** como métrica de selección porque combina precisión y recall, equilibrando:

- **Falsos positivos**: recomendar clientes que luego muestran comportamiento desfavorable.
- **Falsos negativos**: descartar clientes que sí podrían ser buenos.

También se registran Accuracy, ROC-AUC, average precision, precision y recall para observar el comportamiento desde varias perspectivas (`params.yaml` → `train.selection_metric: F1`).

## Resultado de referencia

Random Forest fue el mejor candidato bajo F1:

| Accuracy | Precision | Recall | F1 |
|---------:|----------:|-------:|---:|
| 0,9808 | 0,9567 | 0,9870 | **0,9716** |

Valores reproducidos en `metrics/eval.json`. Detalle comparativo en [Experimentos con MLflow](06-mlflow.md).

## Siguiente lectura

- [2. Conjunto de datos](02-conjunto-datos.md) — EDA, limpieza y variables.
- [3. Estructura](03-estructura.md) — arquitectura end-to-end.
