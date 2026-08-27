# Guía de diapositivas (4–11)

Texto listo para copiar a PowerPoint / Google Slides. Cada bloque = **2 diapositivas**. Ampliar con capturas de [`assets/`](assets/README.md) y figuras de `reports/figures/` tras entrenar.

---

## Diapositivas 4-5 — Arquitectura end-to-end {#diapositivas-4-5}

### Slide 4 — Título: Arquitectura de la solución

**Bullets:**

- Flujo MLOps completo: datos → features → entrenamiento → MLflow → API → Docker
- Estándar de carpetas Cookiecutter Data Science + paquete `caso_berka_model`
- Configuración central: `params.yaml` · pipeline: `dvc.yaml`

**Pie de figura:** *Diagrama end-to-end (pegar Mermaid exportado o screenshot de la doc).*

```text
Datos Berka → DataProcessor → FeatureEngineer → tabla_minable
→ ModelTrainer → Evaluator + MLflow → Registry Production
→ FastAPI → Docker :8000
```

### Slide 5 — Título: Capas y stack

**Bullets:**

| Capa | Tecnología |
|------|------------|
| Datos | DVC + remote local |
| Modelado | scikit-learn (4 candidatos, F1) |
| Experimentos | MLflow Tracking + Registry |
| Serving | FastAPI + Pydantic |
| Runtime | Docker / Compose |

- Separación Git (código/punteros) vs DVC (artefactos pesados)
- Linaje: commit Git + estado DVC como tags en MLflow

**Notas del orador:** Enfatizar reproducibilidad (`dvc repro`) y que la API sirve el modelo **Production**, no un joblib suelto sin gobernanza.

---

## Diapositivas 6-7 — Experimentación MLflow {#diapositivas-6-7}

### Slide 6 — Título: Comparativa de modelos

**Bullets:**

- Experimento: `Berka_Credit_Classification`
- Run padre + 4 nested runs
- Métrica de selección: **F1**

| Modelo | F1 | ROC-AUC |
|--------|---:|--------:|
| Regresión Logística | 0,8829 | 0,9752 |
| KNN | 0,7372 | 0,8861 |
| Árbol de Decisión | 0,9485 | 0,9600 |
| **Random Forest** | **0,9716** | **0,9977** |

**Pie de figura:** *Tabla + `comparacion_metricas.png` / captura UI MLflow.*

### Slide 7 — Título: Modelo en Production

**Bullets:**

- Ganador: **Random Forest** (Accuracy 0,9808 · Precision 0,9567 · Recall 0,9870)
- PyFunc `EnterpriseDecisionWrapper`: umbral **0,65** · alta confianza **0,85**
- Registro: `Berka_BuenCliente` → alias **Production**
- Tags: `git_commit`, `dvc_status`

**Pie de figura:** *Captura Model Registry + ROC del mejor modelo.*

**Notas del orador:** Recordar que `buen_cliente` es heurística; no afirmar detección de fraude.

---

## Diapositivas 8-9 — Pipeline (DVC + Pytest) y API (FastAPI + Docker) {#diapositivas-8-9}

### Slide 8 — Título: Pipeline DVC y tests

**Bullets:**

- Stages: `preprocess` → `train`
- Salidas: `tabla_minable.csv`, `best_model.joblib`, `metrics/eval.json`
- Comandos: `dvc repro` · `dvc metrics show` · `make test`
- Pytest: datos · API (mock) · MLflow (SQLite temporal)

```text
pytest tests   → contratos unitarios + integración aislada
dvc repro      → pipeline al día
```

**Pie de figura:** *Salida de terminal `dvc status` / `pytest` (verde).*

### Slide 9 — Título: API y contenedor

**Bullets:**

- Endpoints: `GET /` · `GET /health` · `POST /predict`
- Contrato: **8 features** preprocesadas (Pydantic)
- Swagger: `/docs`
- Docker: `python:3.11-slim` · modelo en `models/docker_production`
- `make docker-build` → `make docker-run` → `curl /health`

**Pie de figura:** *Screenshot Swagger + respuesta `/health`.*

---

## Diapositivas 10-11 — Buenas prácticas {#diapositivas-10-11}

### Slide 10 — Título: POO y PEP8

**Bullets:**

- Clases por responsabilidad: `DataProcessor`, `FeatureEngineer`, `ModelTrainer`, `Evaluator`, …
- Patrones: Facade (`run`), Strategy (candidatos), Adapter (PyFunc), Registry
- Ruff: línea 99, isort, `make lint` / `make format`
- Params fuera del código (`params.yaml`)

### Slide 11 — Título: GitFlow y colaboración

**Bullets:**

- Ramas: `main` · `develop` · `feature/*` · `hotfix/*`
- PRs + commits pequeños por área (datos, modelo, MLflow, API, docs)
- Checklist pre-merge: `pytest` · `ruff` · `dvc repro` · `/health`
- Equipo Team 8: roles por módulo (ver [GitFlow](10-gitflow.md))

**Pie de figura:** *Diagrama de ramas (exportar Mermaid de la doc 10).*

---

## Checklist de capturas antes de la presentación

1. [ ] MLflow — experimento y nested runs  
2. [ ] MLflow — Registry Production  
3. [ ] Swagger UI `/docs`  
4. [ ] `curl` `/health` o `/predict`  
5. [ ] Terminal `pytest` + `dvc metrics show`  
6. [ ] (Opcional) Figuras ROC / comparación métricas  

Carpeta sugerida: [`docs/docs/assets/`](assets/README.md).
