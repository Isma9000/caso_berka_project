import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import permutation_importance

from caso_berka_model.config import FIGURES_DIR


def preparar_carpeta_figuras():
    """
    Crea la carpeta reports/figures si no existe.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def graficar_roc_mejor_modelo(
    best_model,
    best_model_name,
    X_test,
    y_test,
    mostrar=False
):
    """
    Grafica y guarda la curva ROC del mejor modelo.
    """

    preparar_carpeta_figuras()

    RocCurveDisplay.from_estimator(
        best_model,
        X_test,
        y_test
    )

    plt.title(
        f"Curva ROC del mejor modelo - {best_model_name}"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "roc_mejor_modelo.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )


def graficar_roc_modelos(
    modelos,
    X_test,
    y_test,
    mostrar=False
):
    """
    Compara y guarda las curvas ROC
    de todos los modelos.
    """

    preparar_carpeta_figuras()

    plt.figure(
        figsize=(8, 6)
    )

    for nombre, modelo in modelos.items():

        RocCurveDisplay.from_estimator(
            modelo,
            X_test,
            y_test,
            name=nombre,
            ax=plt.gca()
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title(
        "Curvas ROC de modelos candidatos"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "roc_comparacion_modelos.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )


def graficar_comparacion_metricas(
    resultados_df,
    mostrar=False
):
    """
    Compara y guarda las métricas
    de los modelos.
    """

    preparar_carpeta_figuras()

    metricas = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "avg_precision"
    ]

    resultados_df.set_index(
        "Modelo"
    )[metricas].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        "Comparación de métricas por modelo"
    )

    plt.ylabel(
        "Valor de la métrica"
    )

    plt.xlabel(
        "Modelo"
    )

    plt.xticks(
        rotation=45
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend(
        title="Métricas"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "comparacion_metricas.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )


def calcular_permutation_importance(
    modelo,
    X_test,
    y_test
):
    """
    Calcula la importancia de variables
    mediante permutation importance.
    """

    perm = permutation_importance(
        modelo,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature":
            X_test.columns,

        "importance_mean":
            perm.importances_mean,

        "importance_std":
            perm.importances_std
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance_mean",
            ascending=False
        )
    )

    return importance_df


def graficar_permutation_importance(
    importance_df,
    best_model_name,
    top_n=15,
    mostrar=False
):
    """
    Grafica y guarda las variables
    más importantes.
    """

    preparar_carpeta_figuras()

    top_importance = (
        importance_df
        .head(top_n)
        .sort_values(
            "importance_mean",
            ascending=True
        )
    )

    top_importance.plot(
        kind="barh",
        x="feature",
        y="importance_mean",
        legend=False,
        figsize=(8, 5)
    )

    plt.title(
        f"Permutation Importance - {best_model_name}"
    )

    plt.xlabel(
        "Importancia media"
    )

    plt.ylabel(
        "Variable"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "permutation_importance.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )


def graficar_importancia_random_forest(
    rf,
    columnas,
    mostrar=False
):
    """
    Grafica y guarda la importancia
    interna del Random Forest.
    """

    preparar_carpeta_figuras()

    importancias = pd.DataFrame({
        "Variable":
            columnas,

        "Importancia":
            rf.feature_importances_
    })

    importancias = (
        importancias
        .sort_values(
            by="Importancia",
            ascending=False
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importancias["Variable"],
        importancias["Importancia"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Importancia de variables - Random Forest"
    )

    plt.xlabel(
        "Importancia"
    )

    plt.ylabel(
        "Variable"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "importancia_random_forest.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )


def graficar_ganancia_acumulada(
    decile_table,
    best_model_name,
    mostrar=False
):
    """
    Grafica y guarda la curva
    de ganancia acumulada.
    """

    preparar_carpeta_figuras()

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        decile_table[
            "clientes_acum"
        ],
        decile_table[
            "captura_buen_cliente_acum"
        ],
        marker="o"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title(
        f"Curva de ganancia acumulada - {best_model_name}"
    )

    plt.xlabel(
        "Proporción acumulada de clientes contactados"
    )

    plt.ylabel(
        "Proporción acumulada de buenos clientes capturados"
    )

    plt.tight_layout()

    ruta = (
        FIGURES_DIR
        / "ganancia_acumulada.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(
        f"Gráfico guardado en: {ruta}"
    )