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

DVC queda en el `.venv` tras `pip install`. Activa el entorno (`source .venv/bin/activate`) o usa `make dvc-pull` / `make dvc-repro`.

## 2. Tests rápidos (desarrollo)

No requieren DVC ni datos pesados:

```bash
make ci-local
# o por capas:
make test-unit
make test-integration
```

## 3. Datos (DVC)

```bash
mkdir -p ../dvc_storage_remote
dvc pull
```

`dvc pull` descarga artefactos del remote `../dvc_storage_remote` (definido en `.dvc/config`). No ejecuta el pipeline.

## 4. Pipeline

```bash
dvc repro
# o
make dvc-repro
```

`dvc repro` corre `preprocess` → `train` si algo cambió y regenera `dvc.lock`. Con `mlflow.enabled: true` también escribe `mlflow.db` y promueve el modelo a Production.

## 5. Validación completa

```bash
make test
dvc push    # si cambiaste artefactos y quieres publicarlos
```

## 6. Tests, MLflow y API

```bash
make mlflow-ui          # http://127.0.0.1:5000
make api-serve          # http://127.0.0.1:8000/docs
```

`make mlflow-train` solo hace falta si entrenaste sin MLflow o quieres re-loguear sin `dvc repro`.

## 7. Docker

Requisito previo: modelo en Production (normalmente tras `dvc repro`).

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
