# Caso Banco Berka

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelo predictivo supervisado que permita clasificar clientes buenos y malos para la asignación de tarjetas de crédito.

Realizado por el grupo:

 - Ayala Torrico Adriana Nicole ,
 -  Poma Limache Alisson Daniela,
 -  Fuentes Rios Beatriz,
 -  Peralta Fernández Ismael,
 -  Vargas Orellana José Roberto

Repositorio: https://github.com/Isma9000/caso_berka_project

## Ejecución

### 1. Entorno virtual e instalación

```bash
cd caso_berka_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requisito: Python 3.11 o superior.

### 2. Datos con DVC (remote local)

Los archivos pesados no van en Git; se versionan con DVC en un directorio **fuera del repo**:

```bash
mkdir -p ../dvc_storage_remote
dvc pull
```

El remote por defecto está en [`.dvc/config`](.dvc/config) como `local_remote` → `../dvc_storage_remote`.

### 3. Pipeline completo (recomendado)

Reproduce las etapas `preprocess` y `train` definidas en [`dvc.yaml`](dvc.yaml):

```bash
dvc repro
# equivalente (usa el Python del .venv automáticamente):
make dvc-repro
```

Salida esperada cuando todo está al día:

```text
'data/raw.dvc' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

### 4. Ejecución por etapas

```bash
# Solo preparación de datos → data/processed/tabla_minable.csv
make data

# Solo entrenamiento → models/best_model.joblib + metrics/
make train
```

Equivalente directo:

```bash
python -m caso_berka_model.dataset
python -m caso_berka_model.modeling.train
```

### 5. Tests

```bash
make test
# o
pytest tests
```

### 6. Publicar artefactos en el remote DVC

Tras cambiar datos o re-entrenar:

```bash
dvc push
# o
make dvc-push
```

### 7. Ver métricas y comparar experimentos

```bash
dvc metrics show
dvc params diff
dvc metrics diff HEAD
```

Hiperparámetros en [`params.yaml`](params.yaml). Tras editarlos, vuelve a correr `dvc repro`.

### 8. MLflow (tracking local + Model Registry)

El entrenamiento registra experimentos en **SQLite local** (`mlflow.db`) y artefactos en `mlruns/`. No hace falta un servidor remoto. En Python 3.14 se instala MLflow 3.x (MLflow 2.x exige `pyarrow<20`, sin wheel para 3.14); el registry escribe tanto el *stage* `Production` como el alias `@Production`.

```bash
# Entrenar, loguear métricas/artefactos y promover el PyFunc a Production
make mlflow-train
# equivalente:
python -m caso_berka_model.mlflow_engine.run

# UI en http://127.0.0.1:5000
make mlflow-ui
# equivalente:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Serving REST genérico MLflow del modelo en Production (puerto 8080)
make mlflow-serve

# API FastAPI custom (puerto 8000): carga Berka_BuenCliente@Production
make api-serve
```

La API expone `GET /`, `GET /health` y `POST /predict` (features ya preprocesadas). Docs interactivas en `http://127.0.0.1:8000/docs`.

Ejemplo de inferencia:

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

También puedes lanzar el entry point declarativo:

```bash
mlflow run . --env-manager local
```

Configuración en [`params.yaml`](params.yaml) (`mlflow.*`): experimento `Berka_Credit_Classification`, modelo registrado `Berka_BuenCliente`, umbral de decisión `0.65`. `make train` y el stage DVC `train` también escriben el run si `mlflow.enabled` es `true`.

---

## Git vs DVC: qué va en cada uno

| Tipo | En Git | En DVC (remote local) | Ignorado por `.gitignore` |
|------|--------|------------------------|---------------------------|
| Código (`caso_berka_model/`) | Sí | — | — |
| Config pipeline (`params.yaml`, `dvc.yaml`, `dvc.lock`) | Sí | — | — |
| Puntero de datos (`data/raw.dvc`) | Sí | — | — |
| Datos crudos (`data/raw/**/*.asc`) | No | Sí | `data/.gitignore` + `*.asc` |
| Dataset procesado (`data/processed/tabla_minable.csv`) | No | Sí | `*.csv` |
| Modelo (`models/best_model.joblib`) | No | Sí | `models/.gitignore` |
| Métricas DVC (`metrics/eval.json`, `metrics/plots.csv`) | Sí | — | excepción `!metrics/*` |
| Tracking MLflow (`mlflow.db`, `mlruns/`) | No | — | `mlflow.db`, `mlruns/`, `mlartifacts/` |
| Gráficos (`reports/figures/*.png`) | No | — | `*.png` |
| Reportes CSV (`reports/*.csv`) | No | — | `*.csv` |
| Entorno (`.venv/`) | No | — | `.venv` |
| Cache DVC (`.dvc/cache/`) | No | — | `.dvc/.gitignore` |
| Remote local (`../dvc_storage_remote/`) | No | — | `dvc_storage_remote/` |

Regla práctica: **Git** guarda código, configuración y punteros `.dvc`; **DVC** guarda datos y modelos pesados. Nunca subas `.asc`, CSV de datos ni `.joblib` a Git.

---

## Comprobación de funcionamiento

Tras `make train`, deberías ver algo como:

```text
Dataset: (5369, 21) | predictoras: (5369, 8)
Mejor modelo seleccionado: Random Forest
[ModelTrainer] Modelo guardado en: .../models/best_model.joblib
[Evaluator] Métricas guardadas en: .../metrics/
[MLflow] Run padre=... | mejor=Random Forest | F1=0.97...
[Registry] Modelo 'Berka_BuenCliente' versión 1 asignado al alias Production
```

Verifica que los archivos pesados **no** están en el índice de Git:

```bash
# Deben aparecer como ignorados (salvo metrics/):
git check-ignore -v data/raw/data/trans.asc
git check-ignore -v data/processed/tabla_minable.csv
git check-ignore -v models/best_model.joblib
git check-ignore -v reports/figures/roc_mejor_modelo.png

# Estos SÍ deben poder versionarse en Git:
git check-ignore -v metrics/eval.json   # no debe coincidir (no ignorado)
```

Comandos de salud del proyecto:

```bash
dvc status          # Data and pipelines are up to date.
pytest tests        # tests de datos + tracking MLflow
dvc metrics show    # accuracy ~0.98, f1_score ~0.97
ls mlflow.db        # backend SQLite local tras make mlflow-train
```

---

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── MLproject          <- Entry point declarativo: mlflow run . --env-manager local
├── python_env.yaml    <- Entorno reproducible para MLflow Projects
├── params.yaml        <- Hiperparámetros del pipeline DVC y config MLflow
├── dvc.yaml           <- Stages preprocess y train
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         caso_berka_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── caso_berka_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes caso_berka_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── api                     <- FastAPI: carga Production y /predict
    │   ├── main.py             <- App uvicorn (make api-serve)
    │   ├── model_loader.py     <- Puente al Model Registry
    │   └── schemas.py          <- Contrato de 8 features preprocesadas
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations
    │
    └── mlflow_engine           <- Tracking, PyFunc, evaluate y Model Registry
        ├── lineage.py          <- Hashes Git/DVC para tags de trazabilidad
        ├── artifacts.py        <- Matriz de confusión y curva ROC
        ├── pyfunc.py           <- Wrapper con umbral de decisión
        ├── evaluator.py        <- mlflow.evaluate sobre el modelo sklearn
        ├── trainer.py          <- Nested runs y registro del mejor modelo
        ├── registry.py         <- Promoción a Production
        └── run.py              <- Orquestación: python -m caso_berka_model.mlflow_engine.run
```

--------

