# Caso Banco Berka — Documentación técnica

Modelo predictivo supervisado para clasificar clientes como **buen cliente** o **mal cliente**, en apoyo a la evaluación crediticia y la asignación de tarjetas. El flujo cubre datos versionados (DVC), experimentación (MLflow), API (FastAPI) y despliegue contenerizado (Docker).

!!! warning "Alcance del modelo"
    La etiqueta `buen_cliente` es **heurística** (ingresos, actividad, saldo y morosidad). No representa fraude observado ni sustituye evaluación financiera o normativa.

## Equipo (Team 8)

- Ayala Torrico Adriana Nicole
- Poma Limache Alisson Daniela
- Fuentes Rios Beatriz
- Peralta Fernández Ismael
- Vargas Orellana José Roberto

Repositorio: [github.com/Isma9000/caso_berka_project](https://github.com/Isma9000/caso_berka_project)

## Mapa de la documentación

| Requisito | Página |
|-----------|--------|
| 1. Introducción y objetivos | [01-introduccion](01-introduccion.md) |
| 2. Conjunto de datos (EDA y variables) | [02-conjunto-datos](02-conjunto-datos.md) |
| 3. Estructura y metodología | [03-estructura](03-estructura.md) |
| 4. DVC | [04-dvc](04-dvc.md) |
| 5. POO y PEP8 | [05-poo-pep8](05-poo-pep8.md) |
| 6. MLflow | [06-mlflow](06-mlflow.md) |
| 7. Testing | [07-testing](07-testing.md) |
| 8. FastAPI | [08-fastapi](08-fastapi.md) |
| 9. Docker | [09-docker](09-docker.md) |
| 10. GitFlow | [10-gitflow](10-gitflow.md) |
| Diapositivas 4–11 | [slides](slides.md) |

## Resultado principal

| Métrica | Valor (Random Forest, test) |
|---------|----------------------------:|
| Accuracy | 0,9808 |
| Precision | 0,9567 |
| Recall | 0,9870 |
| **F1** | **0,9716** |

Fuente: `metrics/eval.json` e `INFORME.md` en la raíz del repositorio.

## Cómo leer esta documentación

1. Empieza por [Introducción](01-introduccion.md) si necesitas el planteamiento de negocio.
2. Usa [Estructura](03-estructura.md) para el diagrama end-to-end.
3. Para la presentación oral, copia bullets desde [Guía de diapositivas](slides.md).
4. Para montar el entorno, ve a [Guía rápida](getting-started.md).
