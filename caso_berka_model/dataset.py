import pandas as pd

from caso_berka_model.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

from caso_berka_model.features import (
    crear_base_clientes,
    crear_variables_transacciones,
    crear_variables_ingresos,
    crear_variables_egresos,
    crear_variables_prestamos,
    crear_variables_tarjetas,
    unir_variables,
    limpiar_ids,
    reemplazar_nulos_iniciales,
    crear_variable_objetivo,
    eliminar_duplicados,
    eliminar_columnas_con_muchos_nulos,
    imputar_datos_faltantes,
    aplicar_one_hot_encoding,
    detectar_outliers,
    escalar_variables,
)


def cargar_datos():
    """
    Carga los archivos originales del caso Banco Berka.
    """

    client = pd.read_csv(
        RAW_DATA_DIR / "client.asc",
        sep=";"
    )

    disp = pd.read_csv(
        RAW_DATA_DIR / "disp.asc",
        sep=";"
    )

    card = pd.read_csv(
        RAW_DATA_DIR / "card.asc",
        sep=";"
    )

    account = pd.read_csv(
        RAW_DATA_DIR / "account.asc",
        sep=";"
    )

    loan = pd.read_csv(
        RAW_DATA_DIR / "loan.asc",
        sep=";"
    )

    trans = pd.read_csv(
        RAW_DATA_DIR / "trans.asc",
        sep=";",
        low_memory=False
    )

    return {
        "client": client,
        "disp": disp,
        "card": card,
        "account": account,
        "loan": loan,
        "trans": trans,
    }


def construir_dataset():
    """
    Ejecuta todo el procesamiento del Notebook Parte 1.
    """

    datos = cargar_datos()

    # 1. Crear base principal
    base = crear_base_clientes(
        datos["client"],
        datos["disp"],
        datos["account"]
    )

    # 2. Crear variables de transacciones
    trans_resumen = crear_variables_transacciones(
        datos["trans"]
    )

    ingresos = crear_variables_ingresos(
        datos["trans"]
    )

    egresos = crear_variables_egresos(
        datos["trans"]
    )

    # 3. Crear variables de préstamos
    loan_resumen = crear_variables_prestamos(
        datos["loan"]
    )

    # 4. Crear variables de tarjetas
    card_resumen = crear_variables_tarjetas(
        datos["card"]
    )

    # 5. Unir todas las variables
    base = unir_variables(
        base,
        trans_resumen,
        ingresos,
        egresos,
        loan_resumen,
        card_resumen
    )

    # 6. Limpiar IDs innecesarios
    base = limpiar_ids(base)

    # 7. Reemplazar nulos iniciales
    base = reemplazar_nulos_iniciales(base)

    # 8. Crear variable objetivo
    base = crear_variable_objetivo(base)

    # 9. Eliminar duplicados
    tabla_minable = eliminar_duplicados(base)

    # 10. Eliminar columnas con demasiados nulos
    tabla_minable = (
        eliminar_columnas_con_muchos_nulos(
            tabla_minable
        )
    )

    # 11. Imputar datos faltantes
    tabla_minable = imputar_datos_faltantes(
        tabla_minable
    )

    # 12. One-hot encoding
    tabla_modelo = aplicar_one_hot_encoding(
        tabla_minable
    )

    # 13. Detectar outliers
    resumen_outliers = detectar_outliers(
        tabla_modelo
    )

    print("Outliers detectados:")
    print(resumen_outliers)

    # 14. Escalar variables
    tabla_modelo, scaler = escalar_variables(
        tabla_modelo
    )

    return tabla_modelo, scaler


def guardar_dataset(tabla_modelo):
    """
    Guarda la tabla final procesada.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_salida = (
        PROCESSED_DATA_DIR
        / "tabla_minable.csv"
    )

    tabla_modelo.to_csv(
        ruta_salida,
        index=False
    )

    print(
        f"Dataset guardado en: "
        f"{ruta_salida}"
    )


def main():
    """
    Ejecuta todo el pipeline de preparación.
    """

    tabla_modelo, scaler = construir_dataset()

    guardar_dataset(
        tabla_modelo
    )

    print(
        "Preparación de datos completada."
    )


if __name__ == "__main__":
    main()