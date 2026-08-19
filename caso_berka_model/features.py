import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def crear_base_clientes(client, disp, account):
    """
    Une las tablas client, disp y account.
    """

    base = client.merge(
        disp,
        on="client_id",
        how="left"
    )

    base = base.merge(
        account,
        on="account_id",
        how="left",
        suffixes=("_client", "_account")
    )

    return base


def crear_variables_transacciones(trans):
    """
    Crea variables resumen de transacciones por cuenta.
    """

    trans_resumen = trans.groupby("account_id").agg(
        total_transacciones=("trans_id", "count"),
        saldo_promedio=("balance", "mean"),
    ).reset_index()

    return trans_resumen


def crear_variables_ingresos(trans):
    """
    Calcula ingresos por cuenta.
    """

    ingresos = (
        trans[trans["type"] == "PRIJEM"]
        .groupby("account_id")
        .agg(
            total_ingresos=("amount", "sum"),
            cantidad_ingresos=("amount", "count")
        )
        .reset_index()
    )

    return ingresos


def crear_variables_egresos(trans):
    """
    Calcula egresos por cuenta.
    """

    egresos = (
        trans[trans["type"] == "VYDAJ"]
        .groupby("account_id")
        .agg(
            total_egresos=("amount", "sum"),
            cantidad_egresos=("amount", "count")
        )
        .reset_index()
    )

    return egresos


def crear_variables_prestamos(loan):
    """
    Crea variables relacionadas con préstamos y morosidad.
    """

    loan = loan.copy()

    loan["moroso"] = (
        loan["status"]
        .isin(["B", "D"])
        .astype(int)
    )

    loan_resumen = loan.groupby("account_id").agg(
        tiene_prestamo=("loan_id", "count"),
        monto_prestamo=("amount", "sum"),
        moroso=("moroso", "max")
    ).reset_index()

    return loan_resumen


def crear_variables_tarjetas(card):
    """
    Crea variables relacionadas con tarjetas.
    """

    card_resumen = card.groupby("disp_id").agg(
        tiene_tarjeta=("card_id", "count"),
        tipo_tarjeta=("type", "first")
    ).reset_index()

    return card_resumen


def unir_variables(
    base,
    trans_resumen,
    ingresos,
    egresos,
    loan_resumen,
    card_resumen
):
    """
    Une todas las variables creadas con la base principal.
    """

    base = base.merge(
        trans_resumen,
        on="account_id",
        how="left"
    )

    base = base.merge(
        ingresos,
        on="account_id",
        how="left"
    )

    base = base.merge(
        egresos,
        on="account_id",
        how="left"
    )

    base = base.merge(
        loan_resumen,
        on="account_id",
        how="left"
    )

    base = base.merge(
        card_resumen,
        on="disp_id",
        how="left"
    )

    return base


def limpiar_ids(base):
    """
    Elimina identificadores de distrito que no serán usados.
    """

    base = base.drop(
        columns=[
            "district_id_client",
            "district_id_account"
        ],
        errors="ignore"
    )

    return base


def reemplazar_nulos_iniciales(base):
    """
    Reemplaza valores nulos de variables creadas.
    """

    base = base.copy()

    base["tiene_prestamo"] = (
        base["tiene_prestamo"]
        .fillna(0)
    )

    base["moroso"] = (
        base["moroso"]
        .fillna(0)
    )

    base["tiene_tarjeta"] = (
        base["tiene_tarjeta"]
        .fillna(0)
    )

    columnas_numericas = [
        "total_transacciones",
        "saldo_promedio",
        "total_ingresos",
        "cantidad_ingresos",
        "total_egresos",
        "cantidad_egresos",
        "monto_prestamo"
    ]

    for col in columnas_numericas:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    return base


def crear_variable_objetivo(base):
    """
    Crea la variable buen_cliente.
    """

    base = base.copy()

    base["buen_cliente"] = np.where(
        (
            base["total_ingresos"]
            > base["total_ingresos"].median()
        )
        & (
            base["total_transacciones"]
            > base["total_transacciones"].median()
        )
        & (base["saldo_promedio"] > 0)
        & (base["moroso"] == 0),
        1,
        0
    )

    return base


def eliminar_duplicados(base):
    """
    Conserva una fila por cliente.
    """

    tabla_minable = base.drop_duplicates(
        subset=["client_id"]
    ).copy()

    return tabla_minable


def eliminar_columnas_con_muchos_nulos(
    tabla_minable,
    limite=0.60
):
    """
    Elimina columnas con más del 60% de valores nulos.
    """

    porcentaje_nulos = (
        tabla_minable
        .isnull()
        .mean()
    )

    columnas_a_eliminar = (
        porcentaje_nulos[
            porcentaje_nulos > limite
        ]
        .index
    )

    tabla_minable = tabla_minable.drop(
        columns=columnas_a_eliminar
    )

    return tabla_minable


def imputar_datos_faltantes(tabla_minable):
    """
    Imputa:
    - numéricos con mediana
    - categóricos con moda
    """

    tabla_minable = tabla_minable.copy()

    columnas_numericas = (
        tabla_minable
        .select_dtypes(
            include=["float64", "int64"]
        )
        .columns
    )

    for col in columnas_numericas:
        tabla_minable[col] = (
            tabla_minable[col]
            .fillna(
                tabla_minable[col].median()
            )
        )

    columnas_categoricas = (
        tabla_minable
        .select_dtypes(
            include=["object"]
        )
        .columns
    )

    for col in columnas_categoricas:
        tabla_minable[col] = (
            tabla_minable[col]
            .fillna(
                tabla_minable[col].mode()[0]
            )
        )

    return tabla_minable


def aplicar_one_hot_encoding(tabla_minable):
    """
    Aplica one-hot encoding a type y frequency.
    """

    features_cualitativos = [
        "type",
        "frequency"
    ]

    dummies = pd.get_dummies(
        tabla_minable[
            features_cualitativos
        ]
    )

    columnas_convertir = [
        "type_DISPONENT",
        "type_OWNER",
        "frequency_POPLATEK MESICNE",
        "frequency_POPLATEK PO OBRATU",
        "frequency_POPLATEK TYDNE"
    ]

    for col in columnas_convertir:
        if col in dummies.columns:
            dummies[col] = (
                dummies[col]
                .astype(int)
            )

    tabla_modelo = pd.concat(
        [
            tabla_minable.drop(
                features_cualitativos,
                axis=1
            ),
            dummies
        ],
        axis=1
    )

    return tabla_modelo


def detectar_outliers(tabla_modelo):
    """
    Detecta outliers mediante el método IQR.
    No los elimina.
    """

    variables_outliers = [
        "total_ingresos",
        "total_egresos",
        "total_transacciones",
        "saldo_promedio",
        "monto_prestamo",
        "cantidad_ingresos",
        "cantidad_egresos"
    ]

    resumen_outliers = {}

    for col in variables_outliers:

        if col in tabla_modelo.columns:

            q1 = tabla_modelo[col].quantile(0.25)
            q3 = tabla_modelo[col].quantile(0.75)

            iqr = q3 - q1

            limite_inferior = (
                q1 - 1.5 * iqr
            )

            limite_superior = (
                q3 + 1.5 * iqr
            )

            cantidad_outliers = tabla_modelo[
                (
                    tabla_modelo[col]
                    < limite_inferior
                )
                |
                (
                    tabla_modelo[col]
                    > limite_superior
                )
            ].shape[0]

            resumen_outliers[col] = (
                cantidad_outliers
            )

    return resumen_outliers


def escalar_variables(tabla_modelo):
    """
    Estandariza las variables cuantitativas.
    """

    tabla_modelo = tabla_modelo.copy()

    features_cuantitativos = [
        "total_transacciones",
        "saldo_promedio",
        "total_ingresos",
        "cantidad_ingresos",
        "total_egresos",
        "cantidad_egresos",
        "monto_prestamo"
    ]

    scaler = StandardScaler()

    tabla_modelo[
        features_cuantitativos
    ] = scaler.fit_transform(
        tabla_modelo[
            features_cuantitativos
        ]
    )

    return tabla_modelo, scaler