FROM python:3.13-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar todo el código del proyecto primero
COPY . /app

# Instalar dependencias (ahora sí encontrará el proyecto para la línea '-e .')
RUN pip install --no-cache-dir -r requirements.txt

# Entorno Docker
ENV ENVIRONMENT=docker

# Puerto FastAPI
EXPOSE 8000

# Ejecutar API
CMD ["uvicorn", "caso_berka_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]