import os
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from caso_berka_model.config import PROJ_ROOT


IS_DOCKER = (
    os.environ.get("ENVIRONMENT") == "docker"
)

MODEL_NAME = "Berka_Buen_Cliente_Model"


def get_mlflow_tracking_uri():
    """
    Devuelve la URI de MLflow según el entorno.
    """

    if IS_DOCKER:
        return "sqlite:////app/mlflow.db"

    return f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"


def load_model():
    """
    Carga el modelo según el entorno.

    Docker:
        models/best_model.joblib

    Local:
        MLflow Model Registry
    """

    try:

        print(
            f"[ModelLoader] Entorno: "
            f"{'Docker' if IS_DOCKER else 'Local'}"
        )

        if IS_DOCKER:

            model_path = (
                "/app/models/best_model.joblib"
            )

            print(
                "[ModelLoader] "
                f"Cargando modelo desde: "
                f"{model_path}"
            )

            return joblib.load(
                model_path
            )

        tracking_uri = (
            get_mlflow_tracking_uri()
        )

        mlflow.set_tracking_uri(
            tracking_uri
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        versiones = (
            client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )
        )

        if not versiones:

            raise RuntimeError(
                "No existen versiones "
                f"registradas de {MODEL_NAME}."
            )

        ultima_version = max(
            versiones,
            key=lambda v: int(v.version)
        )

        model_uri = (
            f"models:/{MODEL_NAME}/"
            f"{ultima_version.version}"
        )

        print(
            "[ModelLoader] "
            f"Cargando modelo MLflow: "
            f"{model_uri}"
        )

        return mlflow.sklearn.load_model(
            model_uri
        )

    except Exception as error:

        print(
            "[ModelLoader ERROR] "
            f"{error}"
        )

        return None


def get_model_metadata():
    """
    Devuelve metadatos del modelo.
    """

    if IS_DOCKER:

        return {
            "version": "docker-local",
            "run_id": "joblib"
        }

    try:

        tracking_uri = (
            get_mlflow_tracking_uri()
        )

        mlflow.set_tracking_uri(
            tracking_uri
        )

        client = MlflowClient(
            tracking_uri=tracking_uri
        )

        versiones = (
            client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )
        )

        if versiones:

            ultima_version = max(
                versiones,
                key=lambda v: int(v.version)
            )

            return {
                "version":
                    ultima_version.version,

                "run_id":
                    ultima_version.run_id
            }

    except Exception:
        pass

    return {
        "version": "Desconocida",
        "run_id": "Desconocido"
    }