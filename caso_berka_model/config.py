from pathlib import Path

# Raíz del proyecto
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = PROJ_ROOT / "metrics"

PARAMS_PATH = PROJ_ROOT / "params.yaml"

# Archivo final de preparación
TABLA_MINABLE = PROCESSED_DATA_DIR / "tabla_minable.csv"
