# Imagen oficial y ligera de Python (proyecto requiere >=3.11)
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema necesarias para compilar paquetes nativos
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar el proyecto (incluye models/docker_production tras make docker-prepare-model)
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Indica al model_loader que estamos en el entorno Docker
ENV ENVIRONMENT=docker
ENV DOCKER_MODEL_PATH=/app/models/docker_production

CMD ["uvicorn", "caso_berka_model.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
