from __future__ import annotations

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from caso_berka_model.mlflow_engine.settings import resolve_tracking_uri


class MLflowGovernanceManager:
    """Administra el ciclo de vida de modelos en el Model Registry (stages MLflow 2.x)."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = resolve_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def latest_version(self, model_name: str) -> int:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No hay versiones registradas para '{model_name}'.")
        return max(int(version.version) for version in versions)

    def promote_to_production(self, model_name: str, version: int) -> None:
        version_str = str(version)
        if hasattr(self.client, "transition_model_version_stage"):
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version_str,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(
                    f"[Registry] Modelo '{model_name}' versión {version} "
                    "promovido a Production"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Registry] Stages no disponibles, usando alias: {exc}")

        self.client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=version_str,
        )
        print(
            f"[Registry] Modelo '{model_name}' versión {version} "
            "asignado al alias Production"
        )

    def production_model_uri(self, model_name: str) -> str:
        return f"models:/{model_name}@Production"

    def load_latest_production_model(self, model_name: str):
        uris = [
            f"models:/{model_name}@Production",
            f"models:/{model_name}/Production",
        ]
        last_error: Exception | None = None
        for model_uri in uris:
            try:
                print(f"[Inferencia] Cargando modelo en Producción desde: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"No se pudo cargar '{model_name}' en Production: {last_error}"
        )
