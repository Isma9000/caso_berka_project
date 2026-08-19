import joblib
import pandas as pd

from caso_berka_model.config import MODELS_DIR


def cargar_modelo(
    nombre_archivo="best_model.joblib"
):
    """
    Carga el modelo entrenado.
    """

    ruta_modelo = (
        MODELS_DIR
        / nombre_archivo
    )

    modelo = joblib.load(
        ruta_modelo
    )

    return modelo


def predecir_clientes(
    modelo,
    X
):
    """
    Genera predicción y probabilidad
    de buen cliente.
    """

    prediccion = modelo.predict(
        X
    )

    probabilidades = (
        modelo.predict_proba(X)[:, 1]
    )

    resultados = pd.DataFrame({
        "prediccion":
            prediccion,
        "score_buen_cliente":
            probabilidades
    })

    return resultados


def crear_tabla_deciles(
    modelo,
    X_test,
    y_test
):
    """
    Crea la tabla de deciles
    utilizada para priorización comercial.
    """

    y_prob = (
        modelo
        .predict_proba(
            X_test
        )[:, 1]
    )

    scoring_df = (
        X_test.copy()
    )

    scoring_df[
        "y_real"
    ] = y_test.values

    scoring_df[
        "score_buen_cliente"
    ] = y_prob

    scoring_df[
        "decil_riesgo"
    ] = pd.qcut(
        scoring_df[
            "score_buen_cliente"
        ].rank(
            method="first",
            ascending=False
        ),
        q=10,
        labels=list(
            range(1, 11)
        )
    ).astype(int)

    decile_table = (
        scoring_df
        .groupby(
            "decil_riesgo"
        )
        .agg(
            clientes=(
                "y_real",
                "size"
            ),

            buen_cliente_reales=(
                "y_real",
                "sum"
            ),

            score_min=(
                "score_buen_cliente",
                "min"
            ),

            score_max=(
                "score_buen_cliente",
                "max"
            ),

            score_promedio=(
                "score_buen_cliente",
                "mean"
            ),

            tasa_buen_cliente=(
                "y_real",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "decil_riesgo"
        )
    )

    decile_table[
        "captura_buen_cliente_acum"
    ] = (
        decile_table[
            "buen_cliente_reales"
        ].cumsum()
        /
        decile_table[
            "buen_cliente_reales"
        ].sum()
    )

    decile_table[
        "clientes_acum"
    ] = (
        decile_table[
            "clientes"
        ].cumsum()
        /
        decile_table[
            "clientes"
        ].sum()
    )

    return (
        scoring_df,
        decile_table
    )