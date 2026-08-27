import joblib
import numpy as np
import pandas as pd

from caso_berka_model.modeling.mlflow_tracking import (
    registrar_modelo_mlflow,
)

from caso_berka_model.plots import (
    graficar_roc_mejor_modelo,
    graficar_roc_modelos,
    graficar_comparacion_metricas,
    calcular_permutation_importance,
    graficar_permutation_importance,
    graficar_importancia_random_forest,
    graficar_ganancia_acumulada,
)

from caso_berka_model.modeling.predict import (
    crear_tabla_deciles,
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from caso_berka_model.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR
)


def cargar_dataset():
    """
    Carga la tabla minable generada
    en la etapa de preparación.
    """

    ruta = (
        PROCESSED_DATA_DIR
        / "tabla_minable.csv"
    )

    df = pd.read_csv(ruta)

    return df


def separar_variables(df):
    """
    Separa la variable objetivo de las variables predictoras.

    Se eliminan también las variables utilizadas directamente
    para construir 'buen_cliente', evitando data leakage.
    """

    y = df["buen_cliente"]

    columnas_eliminar = [
        # Variable objetivo
        "buen_cliente",

        # Identificadores
        "client_id",
        "disp_id",
        "account_id",

        # Variables utilizadas directamente
        # para crear buen_cliente
        "total_ingresos",
        "total_transacciones",
        "saldo_promedio",
        "moroso",

        # Variables categóricas eliminadas
        # en el notebook original
        "type_DISPONENT",
        "type_OWNER",
        "frequency_POPLATEK MESICNE",
        "frequency_POPLATEK PO OBRATU",
        "frequency_POPLATEK TYDNE"
    ]

    X = df.drop(
        columns=columnas_eliminar,
        errors="ignore"
    )

    return X, y


def dividir_datos(X, y):
    """
    Divide los datos en entrenamiento
    y prueba utilizando estratificación.
    """

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def entrenar_regresion_logistica(
    X_train,
    y_train
):
    """
    Entrena una Regresión Logística Binaria.
    """

    modelo = make_pipeline(
        #StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def buscar_mejor_k(X, y):
    """
    Busca el mejor valor de k para KNN
    mediante validación cruzada.
    """

    k_range = range(1, 21)

    cv_scores = []

    for k in k_range:

        knn = KNeighborsClassifier(
            n_neighbors=k
        )

        scores = cross_val_score(
            knn,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        cv_scores.append(
            scores.mean()
        )

    best_k = list(k_range)[
        np.argmax(cv_scores)
    ]

    return best_k, cv_scores


def entrenar_knn(
    X_train,
    y_train,
    best_k
):
    """
    Entrena el modelo KNN
    usando el mejor valor de k.
    """

    modelo = make_pipeline(
        #StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=best_k,
            weights="distance"
        )
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def entrenar_arbol(
    X_train,
    y_train
):
    """
    Entrena un Árbol de Decisión.
    """

    modelo = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        max_depth=None,
        class_weight="balanced"
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def entrenar_random_forest(
    X_train,
    y_train
):
    """
    Entrena un Random Forest.
    """

    modelo = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo


def evaluar_modelo(
    nombre,
    modelo,
    X_test,
    y_test
):
    """
    Calcula las métricas
    utilizadas en el Notebook 2.
    """

    y_pred = modelo.predict(
        X_test
    )

    y_prob = modelo.predict_proba(
        X_test
    )[:, 1]

    resultado = {
        "Modelo": nombre,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "avg_precision":
            average_precision_score(
                y_test,
                y_prob
            ),

        "Matriz_Confusion":
            confusion_matrix(
                y_test,
                y_pred
            )
    }

    return resultado


def evaluar_modelos(
    modelos,
    X_test,
    y_test
):
    """
    Evalúa todos los modelos.
    """

    resultados = []

    for nombre, modelo in modelos.items():

        resultado = evaluar_modelo(
            nombre,
            modelo,
            X_test,
            y_test
        )

        resultados.append(
            resultado
        )

    resultados_df = pd.DataFrame(
        resultados
    )

    return resultados_df


def guardar_modelo(
    modelo,
    nombre_archivo="best_model.joblib"
):
    """
    Guarda el modelo seleccionado.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta = (
        MODELS_DIR
        / nombre_archivo
    )

    joblib.dump(
        modelo,
        ruta
    )

    print(
        f"Modelo guardado en: {ruta}"
    )


def main():
    """
    Ejecuta todo el pipeline de modelado.
    """

    # 1. Cargar dataset
    df = cargar_dataset()

    print(
        f"Dataset cargado: {df.shape}"
    )

    # 2. Separar X e y
    X, y = separar_variables(
        df
    )

    print(
        f"Variables predictoras: {X.shape}"
    )

    # 3. División train/test
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = dividir_datos(
        X,
        y
    )

    # 4. Regresión Logística
    print(
        "\nEntrenando Regresión Logística..."
    )

    rlb = entrenar_regresion_logistica(
        X_train,
        y_train
    )

    # 5. Buscar mejor k
    print(
        "\nBuscando mejor k para KNN..."
    )

    best_k, cv_scores = buscar_mejor_k(
        X,
        y
    )

    print(
        f"Mejor k encontrado: {best_k}"
    )

    # 6. KNN
    print(
        "\nEntrenando KNN..."
    )

    knn = entrenar_knn(
        X_train,
        y_train,
        best_k
    )

    # 7. Árbol de Decisión
    print(
        "\nEntrenando Árbol de Decisión..."
    )

    arbol = entrenar_arbol(
        X_train,
        y_train
    )

    # 8. Random Forest
    print(
        "\nEntrenando Random Forest..."
    )

    rf = entrenar_random_forest(
        X_train,
        y_train
    )

    # 9. Agrupar modelos
    modelos = {
        "Regresión Logística": rlb,
        "KNN": knn,
        "Árbol de Decisión": arbol,
        "Random Forest": rf
    }

        # ----------------------------------
    # REGISTRAR EXPERIMENTOS EN MLFLOW
    # ----------------------------------

    registrar_modelo_mlflow(
        nombre="Regresion_Logistica",
        modelo=rlb,
        X_test=X_test,
        y_test=y_test,
        parametros={
            "max_iter": 1000,
            "class_weight": "balanced"
        }
    )

    registrar_modelo_mlflow(
        nombre="KNN",
        modelo=knn,
        X_test=X_test,
        y_test=y_test,
        parametros={
            "n_neighbors": best_k,
            "weights": "distance"
        }
    )

    registrar_modelo_mlflow(
        nombre="Arbol_Decision",
        modelo=arbol,
        X_test=X_test,
        y_test=y_test,
        parametros={
            "criterion": "gini",
            "max_depth": "None",
            "class_weight": "balanced"
        }
    )

    registrar_modelo_mlflow(
        nombre="Random_Forest",
        modelo=rf,
        X_test=X_test,
        y_test=y_test,
        parametros={
            "n_estimators": 400,
            "max_depth": "None",
            "min_samples_leaf": 3,
            "class_weight":
                "balanced_subsample"
        }
    )

    # 10. Evaluación
    resultados_df = evaluar_modelos(
        modelos,
        X_test,
        y_test
    )

    print(
        "\nResultados:"
    )

    print(
        resultados_df[
            [
                "Modelo",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "avg_precision"
            ]
        ]
    )

    # 11. Selección del modelo
    #
    # En tu Notebook 2 elegiste
    # Árbol de Decisión por su
    # capacidad de interpretación.
    #best_model_name = (
    #    "Árbol de Decisión"
    #)

    best_model_name = resultados_df.loc[
    resultados_df["F1"].idxmax(),
    "Modelo"
    ]

    best_model = modelos[
        best_model_name
    ]

    print(
        "\nMejor modelo seleccionado:"
    )

    print(
        best_model_name
    )

    # Registrar el mejor modelo
    # también en Model Registry
    registrar_modelo_mlflow(
        nombre=f"BEST_{best_model_name}",
        modelo=best_model,
        X_test=X_test,
        y_test=y_test,
        parametros={
            "selection_metric": "F1"
        },
        registrar_modelo=True
    )


    # 12. Guardar modelo
    guardar_modelo(
        best_model
    )

        # 13. Curva ROC del mejor modelo
    graficar_roc_mejor_modelo(
        best_model,
        best_model_name,
        X_test,
        y_test
    )

    # 14. Comparación ROC de todos los modelos
    graficar_roc_modelos(
        modelos,
        X_test,
        y_test
    )

    # 15. Comparación visual de métricas
    graficar_comparacion_metricas(
        resultados_df
    )

    # 16. Permutation Importance
    importance_df = (
        calcular_permutation_importance(
            best_model,
            X_test,
            y_test
        )
    )

    print(
        "\nImportancia de variables:"
    )

    print(
        importance_df.head(15)
    )

    graficar_permutation_importance(
        importance_df,
        best_model_name
    )

    # 17. Importancia del Random Forest
    graficar_importancia_random_forest(
        rf,
        X.columns
    )

    # 18. Tabla de deciles
    scoring_df, decile_table = (
        crear_tabla_deciles(
            best_model,
            X_test,
            y_test
        )
    )

    print(
        "\nTabla de deciles:"
    )

    print(
        decile_table
    )

    # 19. Curva de ganancia acumulada
    graficar_ganancia_acumulada(
        decile_table,
        best_model_name
    )

        # Crear carpeta reports si no existe
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar métricas
    resultados_df.drop(
        columns=["Matriz_Confusion"],
        errors="ignore"
    ).to_csv(
        REPORTS_DIR / "metricas_modelos.csv",
        index=False
    )

    # Guardar importancia de variables
    importance_df.to_csv(
        REPORTS_DIR / "importancia_variables.csv",
        index=False
    )

    # Guardar tabla de deciles
    decile_table.to_csv(
        REPORTS_DIR / "tabla_deciles.csv",
        index=False
    )


if __name__ == "__main__":
    main()