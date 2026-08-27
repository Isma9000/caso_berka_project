import subprocess

import mlflow
import mlflow.sklearn

from caso_berka_model.config import (
    FIGURES_DIR,
    PROJ_ROOT,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


TRACKING_URI = (
    f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
)

EXPERIMENT_NAME = "Caso_Berka_Clasificacion"

REGISTERED_MODEL_NAME = "Berka_Buen_Cliente_Model"


def configurar_mlflow():
    """
    Configura MLflow para utilizar una base SQLite local.
    """

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def obtener_version_git():
    """
    Obtiene el hash del commit actual para trazabilidad.
    """

    try:

        resultado = subprocess.run(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD"
            ],
            capture_output=True,
            text=True,
            check=False
        )

        hash_git = resultado.stdout.strip()

        if hash_git:
            return hash_git

    except Exception:
        pass

    return "sin-version-git"


def obtener_estado_dvc():
    """
    Comprueba el estado actual de DVC.
    """

    try:

        resultado = subprocess.run(
            [
                "dvc",
                "status"
            ],
            capture_output=True,
            text=True,
            check=False
        )

        salida = resultado.stdout.strip()

        if not salida:
            return "up-to-date"

        if "Data and pipelines are up to date" in salida:
            return "up-to-date"

        return "modified"

    except Exception:

        return "sin-dvc"


def calcular_metricas(
    modelo,
    X_test,
    y_test
):
    """
    Calcula las métricas del modelo.
    """

    y_pred = modelo.predict(
        X_test
    )

    y_prob = modelo.predict_proba(
        X_test
    )[:, 1]

    metricas = {

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                y_prob
            ),

        "average_precision":
            average_precision_score(
                y_test,
                y_prob
            ),
    }

    return metricas

def registrar_graficos_mlflow():
    """
    Registra los gráficos existentes
    de reports/figures como artefactos.
    """

    if not FIGURES_DIR.exists():
        return

    for archivo in FIGURES_DIR.glob("*.png"):

        mlflow.log_artifact(
            str(archivo),
            artifact_path="charts"
        )

def registrar_modelo_mlflow(
    nombre,
    modelo,
    X_test,
    y_test,
    parametros=None,
    registrar_modelo=False
):
    """
    Registra una ejecución completa en MLflow.
    """

    configurar_mlflow()

    version_git = obtener_version_git()
    estado_dvc = obtener_estado_dvc()

    with mlflow.start_run(
        run_name=nombre
    ) as run:

        # -------------------------
        # TAGS
        # -------------------------

        mlflow.set_tags({

            "project":
                "Caso Banco Berka",

            "framework":
                "scikit-learn",

            "git_version":
                version_git,

            "dvc_status":
                estado_dvc,

            "model_type":
                nombre
        })

        # -------------------------
        # PARÁMETROS
        # -------------------------

        if parametros:

            mlflow.log_params(
                parametros
            )

        # -------------------------
        # MÉTRICAS
        # -------------------------

        metricas = calcular_metricas(
            modelo,
            X_test,
            y_test
        )

        mlflow.log_metrics(
            metricas
        )

        # -------------------------
        # GRÁFICOS
        # -------------------------

        if registrar_modelo:
            registrar_graficos_mlflow()

        # -------------------------
        # MODELO
        # -------------------------

        input_example = (
            X_test.iloc[:5]
        )
        

        if registrar_modelo:

            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=(
                    REGISTERED_MODEL_NAME
                ),
                serialization_format="cloudpickle",
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="model",
                input_example=input_example,
                serialization_format="cloudpickle",
            )

        print(
            f"MLflow Run registrado: "
            f"{nombre}"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        return run.info.run_id
