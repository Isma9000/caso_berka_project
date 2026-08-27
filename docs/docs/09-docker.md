# 9. Contenerización con Docker

## Motivación

Aislar el runtime de la API (Python 3.11, dependencias y modelo Production) para ejecutar el servicio sin depender del entorno local del desarrollador.

## Dockerfile

Base: `python:3.11-slim`

Pasos relevantes:

1. `WORKDIR /app`
2. Instalar `build-essential` y `curl` (compilación de wheels nativos)
3. `COPY` del proyecto e `pip install -r requirements.txt`
4. `EXPOSE 8000`
5. Variables: `ENVIRONMENT=docker`, `DOCKER_MODEL_PATH=/app/models/docker_production`
6. `CMD`: `uvicorn caso_berka_model.api.main:app --host 0.0.0.0 --port 8000`

## Por qué `models/docker_production`

El Model Registry local guarda **rutas absolutas del host** que no existen dentro del contenedor. Antes del build se copia el artefacto PyFunc de Production:

```bash
make docker-prepare-model
# python -m caso_berka_model.api.prepare_docker_model
```

Eso escribe:

- `models/docker_production/` — artefacto MLflow
- `models/docker_meta.env` — `DOCKER_MODEL_VERSION`, `DOCKER_RUN_ID`, etc.

Requisito previo: `mlflow.db` con un modelo en Production (`make mlflow-train`).

## Docker Compose

Servicio `api` en `docker-compose.yml`:

- Imagen `caso-berka-api`
- Puertos `8000:8000`
- `env_file: models/docker_meta.env`
- `ENVIRONMENT=docker`

## Levantamiento

```bash
make docker-build    # prepare-model + docker build -t caso-berka-api
make docker-run      # -p 8000:8000 + env-file

# Alternativa
docker compose up --build
```

Comprobación:

```bash
curl -s http://127.0.0.1:8000/health
```

Tras promover una **nueva** versión a Production, volver a ejecutar `make docker-build` (o al menos `make docker-prepare-model` y rebuild).

## `.dockerignore`

Excluye datos crudos, cache DVC, `mlflow.db`/`mlruns`, notebooks y el joblib local; la imagen lleva el PyFunc preparado, no el registry del host.

## Siguiente lectura

- [10. GitFlow](10-gitflow.md)
- [Diapositivas 8–9](slides.md#diapositivas-8-9)
