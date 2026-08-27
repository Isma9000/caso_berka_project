import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Transformaciones de tablas Berka hacia la tabla minable."""

    def crear_base_clientes(self, client, disp, account):
        base = client.merge(disp, on="client_id", how="left")
        base = base.merge(
            account,
            on="account_id",
            how="left",
            suffixes=("_client", "_account"),
        )
        return base

    def crear_variables_transacciones(self, trans):
        return (
            trans.groupby("account_id")
            .agg(
                total_transacciones=("trans_id", "count"),
                saldo_promedio=("balance", "mean"),
            )
            .reset_index()
        )

    def crear_variables_ingresos(self, trans):
        return (
            trans[trans["type"] == "PRIJEM"]
            .groupby("account_id")
            .agg(
                total_ingresos=("amount", "sum"),
                cantidad_ingresos=("amount", "count"),
            )
            .reset_index()
        )

    def crear_variables_egresos(self, trans):
        return (
            trans[trans["type"] == "VYDAJ"]
            .groupby("account_id")
            .agg(
                total_egresos=("amount", "sum"),
                cantidad_egresos=("amount", "count"),
            )
            .reset_index()
        )

    def crear_variables_prestamos(self, loan):
        loan = loan.copy()
        loan["moroso"] = loan["status"].isin(["B", "D"]).astype(int)
        return (
            loan.groupby("account_id")
            .agg(
                tiene_prestamo=("loan_id", "count"),
                monto_prestamo=("amount", "sum"),
                moroso=("moroso", "max"),
            )
            .reset_index()
        )

    def crear_variables_tarjetas(self, card):
        return (
            card.groupby("disp_id")
            .agg(
                tiene_tarjeta=("card_id", "count"),
                tipo_tarjeta=("type", "first"),
            )
            .reset_index()
        )

    def unir_variables(
        self,
        base,
        trans_resumen,
        ingresos,
        egresos,
        loan_resumen,
        card_resumen,
    ):
        base = base.merge(trans_resumen, on="account_id", how="left")
        base = base.merge(ingresos, on="account_id", how="left")
        base = base.merge(egresos, on="account_id", how="left")
        base = base.merge(loan_resumen, on="account_id", how="left")
        base = base.merge(card_resumen, on="disp_id", how="left")
        return base

    def limpiar_ids(self, base):
        return base.drop(
            columns=["district_id_client", "district_id_account"],
            errors="ignore",
        )

    def reemplazar_nulos_iniciales(self, base):
        base = base.copy()
        for col in ["tiene_prestamo", "moroso", "tiene_tarjeta"]:
            if col in base.columns:
                base[col] = base[col].fillna(0)

        columnas_numericas = [
            "total_transacciones",
            "saldo_promedio",
            "total_ingresos",
            "cantidad_ingresos",
            "total_egresos",
            "cantidad_egresos",
            "monto_prestamo",
        ]
        for col in columnas_numericas:
            if col in base.columns:
                base[col] = base[col].fillna(0)
        return base

    def crear_variable_objetivo(self, base):
        base = base.copy()
        base["buen_cliente"] = np.where(
            (base["total_ingresos"] > base["total_ingresos"].median())
            & (base["total_transacciones"] > base["total_transacciones"].median())
            & (base["saldo_promedio"] > 0)
            & (base["moroso"] == 0),
            1,
            0,
        )
        return base

    def eliminar_duplicados(self, base):
        return base.drop_duplicates(subset=["client_id"]).copy()

    def eliminar_columnas_con_muchos_nulos(self, tabla_minable, limite=0.60):
        porcentaje_nulos = tabla_minable.isnull().mean()
        columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > limite].index
        return tabla_minable.drop(columns=columnas_a_eliminar)

    def imputar_datos_faltantes(self, tabla_minable):
        tabla_minable = tabla_minable.copy()
        columnas_numericas = tabla_minable.select_dtypes(
            include=["float64", "int64"]
        ).columns
        for col in columnas_numericas:
            tabla_minable[col] = tabla_minable[col].fillna(tabla_minable[col].median())

        columnas_categoricas = tabla_minable.select_dtypes(include=["object"]).columns
        for col in columnas_categoricas:
            moda = tabla_minable[col].mode()
            if len(moda) > 0:
                tabla_minable[col] = tabla_minable[col].fillna(moda[0])
        return tabla_minable

    def aplicar_one_hot_encoding(self, tabla_minable):
        features_cualitativos = ["type", "frequency"]
        presentes = [c for c in features_cualitativos if c in tabla_minable.columns]
        if not presentes:
            return tabla_minable

        dummies = pd.get_dummies(tabla_minable[presentes])
        columnas_convertir = [
            "type_DISPONENT",
            "type_OWNER",
            "frequency_POPLATEK MESICNE",
            "frequency_POPLATEK PO OBRATU",
            "frequency_POPLATEK TYDNE",
        ]
        for col in columnas_convertir:
            if col in dummies.columns:
                dummies[col] = dummies[col].astype(int)

        return pd.concat(
            [tabla_minable.drop(presentes, axis=1), dummies],
            axis=1,
        )

    def detectar_outliers(self, tabla_modelo):
        variables_outliers = [
            "total_ingresos",
            "total_egresos",
            "total_transacciones",
            "saldo_promedio",
            "monto_prestamo",
            "cantidad_ingresos",
            "cantidad_egresos",
        ]
        resumen_outliers = {}
        for col in variables_outliers:
            if col not in tabla_modelo.columns:
                continue
            q1 = tabla_modelo[col].quantile(0.25)
            q3 = tabla_modelo[col].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            cantidad_outliers = tabla_modelo[
                (tabla_modelo[col] < limite_inferior)
                | (tabla_modelo[col] > limite_superior)
            ].shape[0]
            resumen_outliers[col] = cantidad_outliers
        return resumen_outliers

    def escalar_variables(self, tabla_modelo):
        tabla_modelo = tabla_modelo.copy()
        features_cuantitativos = [
            "total_transacciones",
            "saldo_promedio",
            "total_ingresos",
            "cantidad_ingresos",
            "total_egresos",
            "cantidad_egresos",
            "monto_prestamo",
        ]
        presentes = [c for c in features_cuantitativos if c in tabla_modelo.columns]
        scaler = StandardScaler()
        tabla_modelo[presentes] = scaler.fit_transform(tabla_modelo[presentes])
        return tabla_modelo, scaler

    def clean_data(self, datos, null_threshold=0.60):
        """Construye la tabla minable a partir de las tablas crudas."""
        base = self.crear_base_clientes(
            datos["client"], datos["disp"], datos["account"]
        )
        trans_resumen = self.crear_variables_transacciones(datos["trans"])
        ingresos = self.crear_variables_ingresos(datos["trans"])
        egresos = self.crear_variables_egresos(datos["trans"])
        loan_resumen = self.crear_variables_prestamos(datos["loan"])
        card_resumen = self.crear_variables_tarjetas(datos["card"])

        base = self.unir_variables(
            base, trans_resumen, ingresos, egresos, loan_resumen, card_resumen
        )
        base = self.limpiar_ids(base)
        base = self.reemplazar_nulos_iniciales(base)
        base = self.crear_variable_objetivo(base)
        tabla_minable = self.eliminar_duplicados(base)
        tabla_minable = self.eliminar_columnas_con_muchos_nulos(
            tabla_minable, limite=null_threshold
        )
        tabla_minable = self.imputar_datos_faltantes(tabla_minable)
        tabla_modelo = self.aplicar_one_hot_encoding(tabla_minable)
        self.detectar_outliers(tabla_modelo)
        tabla_modelo, _scaler = self.escalar_variables(tabla_modelo)
        return tabla_modelo
