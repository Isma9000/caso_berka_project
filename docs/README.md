# Documentación — Caso Banco Berka

Documentación técnica del proyecto MLOps (clasificación de clientes del Banco Berka),
alineada con los 10 requisitos del informe y con una guía de diapositivas.

## Requisitos

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

Incluye `mkdocs` y `mkdocs-material`.

## Servir en local

El archivo de configuración está en esta carpeta (`docs/mkdocs.yml`):

```bash
cd docs
mkdocs serve
```

Abrir [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Construir el sitio estático

```bash
cd docs
mkdocs build
```

La salida queda en `docs/site/` (ignorada por Git).

## Estructura

| Archivo | Contenido |
|---------|-----------|
| `docs/index.md` | Portada y mapa de la documentación |
| `docs/01-…` … `10-…` | Un capítulo por requisito del informe |
| `docs/slides.md` | Texto listo para diapositivas 4–11 |
| `docs/assets/` | Carpeta para capturas (MLflow, Swagger, etc.) |
| `docs/getting-started.md` | Puesta en marcha rápida |

La fuente narrativa principal sigue siendo [`INFORME.md`](../INFORME.md) en la raíz del repo.
