# Guía rápida

Pasos mínimos para reproducir el pipeline en una máquina limpia. Detalle completo en el `README.md` de la raíz del repositorio.

## 1. Entorno

```bash
cd caso_berka_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requisito: **Python 3.11+**.

## 2. Datos (DVC)

```bash
mkdir -p ../dvc_storage_remote   # o la ruta del remote en .dvc/config
dvc pull
```

## 3. Pipeline

```bash
dvc repro
# o
make dvc-repro
```

## 4. Tests, MLflow y API

```bash
make test
make mlflow-ui          # http://127.0.0.1:5000
make api-serve          # http://127.0.0.1:8000/docs
```

## 5. Docker

```bash
make docker-build
make docker-run
curl -s http://127.0.0.1:8000/health
```

## Documentación MkDocs

```bash
cd docs
mkdocs serve
```

Más contexto técnico: capítulos [01](01-introduccion.md)–[10](10-gitflow.md) y [slides](slides.md).
