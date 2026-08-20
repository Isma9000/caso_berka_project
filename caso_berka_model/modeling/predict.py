import joblib
import pandas as pd

from caso_berka_model.config import MODELS_DIR


class Predictor:
    """Carga el modelo entrenado y genera predicciones."""

    def __init__(self, model_path=None, nombre_archivo="best_model.joblib"):
        self.model_path = model_path or (MODELS_DIR / nombre_archivo)
        self.model = None

    def load_model(self, nombre_archivo="best_model.joblib"):
        ruta = self.model_path
        if nombre_archivo != "best_model.joblib":
            ruta = MODELS_DIR / nombre_archivo
        self.model = joblib.load(ruta)
        return self.model

    def predict(self, X):
        if self.model is None:
            self.load_model()
        prediccion = self.model.predict(X)
        probabilidades = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "prediccion": prediccion,
                "score_buen_cliente": probabilidades,
            }
        )

    def decile_table(self, X_test, y_test):
        if self.model is None:
            self.load_model()

        y_prob = self.model.predict_proba(X_test)[:, 1]
        scoring_df = X_test.copy()
        scoring_df["y_real"] = y_test.values
        scoring_df["score_buen_cliente"] = y_prob
        scoring_df["decil_riesgo"] = pd.qcut(
            scoring_df["score_buen_cliente"].rank(method="first", ascending=False),
            q=10,
            labels=list(range(1, 11)),
        ).astype(int)

        decile_table = (
            scoring_df.groupby("decil_riesgo")
            .agg(
                clientes=("y_real", "size"),
                buen_cliente_reales=("y_real", "sum"),
                score_min=("score_buen_cliente", "min"),
                score_max=("score_buen_cliente", "max"),
                score_promedio=("score_buen_cliente", "mean"),
                tasa_buen_cliente=("y_real", "mean"),
            )
            .reset_index()
            .sort_values("decil_riesgo")
        )

        decile_table["captura_buen_cliente_acum"] = (
            decile_table["buen_cliente_reales"].cumsum()
            / decile_table["buen_cliente_reales"].sum()
        )
        decile_table["clientes_acum"] = (
            decile_table["clientes"].cumsum() / decile_table["clientes"].sum()
        )
        return scoring_df, decile_table
