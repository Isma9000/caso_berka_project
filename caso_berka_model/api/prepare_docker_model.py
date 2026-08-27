"""Copia el PyFunc Production a models/docker_production para el build Docker."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3

from caso_berka_model.config import MODELS_DIR, PROJ_ROOT
from caso_berka_model.mlflow_engine.settings import DEFAULT_REGISTERED_MODEL_NAME

DEST_DIR = MODELS_DIR / "docker_production"
META_ENV = MODELS_DIR / "docker_meta.env"
MLFLOW_DB = PROJ_ROOT / "mlflow.db"


def _resolve_production(conn: sqlite3.Connection, model_name: str):
    row = conn.execute(
        """
        SELECT mv.version, mv.run_id, mv.storage_location
        FROM registered_model_aliases a
        JOIN model_versions mv ON mv.name = a.name AND mv.version = a.version
        WHERE a.alias = 'Production' AND a.name = ?
        """,
        (model_name,),
    ).fetchone()
    if row is not None:
        return row

    return conn.execute(
        """
        SELECT version, run_id, storage_location
        FROM model_versions
        WHERE name = ? AND current_stage = 'Production'
        ORDER BY CAST(version AS INTEGER) DESC
        LIMIT 1
        """,
        (model_name,),
    ).fetchone()


def main() -> None:
    if not MLFLOW_DB.exists():
        raise SystemExit("No existe mlflow.db. Ejecuta make mlflow-train primero.")

    with sqlite3.connect(MLFLOW_DB) as conn:
        row = _resolve_production(conn, DEFAULT_REGISTERED_MODEL_NAME)

    if row is None:
        raise SystemExit(
            f"No hay modelo Production para '{DEFAULT_REGISTERED_MODEL_NAME}'. "
            "Ejecuta make mlflow-train."
        )

    version, run_id, storage = row
    src = Path(storage)
    if not src.is_dir():
        raise SystemExit(f"No existe el artefacto Production: {src}")

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    shutil.copytree(src, DEST_DIR)

    META_ENV.write_text(
        (
            f"DOCKER_MODEL_VERSION={version}\n"
            f"DOCKER_RUN_ID={run_id}\n"
            "DOCKER_MODEL_PATH=/app/models/docker_production\n"
            "ENVIRONMENT=docker\n"
        ),
        encoding="utf-8",
    )
    print(f"[Docker] Production v{version} (run_id={run_id}) -> {DEST_DIR}")


if __name__ == "__main__":
    main()
