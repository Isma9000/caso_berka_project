import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import yaml

from caso_berka_model.config import (
    METRICS_DIR,
    MODELS_DIR,
    PARAMS_PATH,
    REPORTS_DIR,
    TABLA_MINABLE,
)
from caso_berka_model.modeling.predict import Predictor
from caso_berka_model.plots import (
    calcular_permutation_importance,
    graficar_comparacion_metricas,
    graficar_ganancia_acumulada,
    graficar_importancia_random_forest,
    graficar_permutation_importance,
    graficar_roc_mejor_modelo,
    graficar_roc_modelos,
)


class Evaluator:
    """Métricas y artefactos de evaluación (JSON/CSV para DVC)."""

    def __init__(self, metrics_dir: str | Path):
        self.metrics_dir = Path(metrics_dir)

    def evaluar_modelo(self, nombre, modelo, X_test, y_test):
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        return {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "avg_precision": average_precision_score(y_test, y_prob),
            "Matriz_Confusion": confusion_matrix(y_test, y_pred),
        }

    def evaluar_modelos(self, modelos, X_test, y_test):
        resultados = [
            self.evaluar_modelo(nombre, modelo, X_test, y_test)
            for nombre, modelo in modelos.items()
        ]
        return pd.DataFrame(resultados)

    def save_dvc_metrics(self, modelo, X_test, y_test, resultado: dict):
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        preds = modelo.predict(X_test)
        metrics = {
            "accuracy": float(resultado["Accuracy"]),
            "precision": float(resultado["Precision"]),
            "recall": float(resultado["Recall"]),
            "f1_score": float(resultado["F1"]),
        }
        with open(self.metrics_dir / "eval.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=4)

        plots_df = pd.DataFrame({"actual": y_test, "predicted": preds})
        plots_df.to_csv(self.metrics_dir / "plots.csv", index=False)
        print(f"[Evaluator] Métricas guardadas en: {self.metrics_dir}/")


class ModelTrainer:
    """Entrenamiento, selección y persistencia del modelo."""

    def __init__(
        self,
        params_path: str | Path | None = None,
        data_path: str | Path | None = None,
        metrics_dir: str | Path | None = None,
    ):
        self.params_path = Path(params_path) if params_path else PARAMS_PATH
        self.data_path = Path(data_path) if data_path else TABLA_MINABLE
        self.metrics_dir = Path(metrics_dir) if metrics_dir else METRICS_DIR
        self.params = self._load_params(self.params_path)
        self.model = None
        self.evaluator = Evaluator(self.metrics_dir)
        self.predictor = Predictor()

    def _load_params(self, params_path: Path) -> dict:
        with open(params_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def cargar_dataset(self):
        return pd.read_csv(self.data_path)

    def separar_variables(self, df):
        target_col = self.params["train"]["target_col"]
        y = df[target_col]
        columnas_eliminar = [
            target_col,
            "client_id",
            "disp_id",
            "account_id",
            "total_ingresos",
            "total_transacciones",
            "saldo_promedio",
            "moroso",
            "type_DISPONENT",
            "type_OWNER",
            "frequency_POPLATEK MESICNE",
            "frequency_POPLATEK PO OBRATU",
            "frequency_POPLATEK TYDNE",
        ]
        X = df.drop(columns=columnas_eliminar, errors="ignore")
        return X, y

    def prepare_data(self):
        df = self.cargar_dataset()
        X, y = self.separar_variables(df)
        prepare = self.params["prepare"]
        return train_test_split(
            X,
            y,
            test_size=prepare["split_ratio"],
            random_state=prepare["random_state"],
            stratify=y,
        )

    def buscar_mejor_k(self, X, y):
        k_max = int(self.params["train"].get("knn_k_max", 20))
        k_range = range(1, k_max + 1)
        cv_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
            cv_scores.append(scores.mean())
        best_k = list(k_range)[np.argmax(cv_scores)]
        return best_k, cv_scores

    def _entrenar_modelos(self, X_train, y_train, X, y):
        train = self.params["train"]
        random_state = self.params["prepare"]["random_state"]

        rlb = make_pipeline(
            LogisticRegression(
                max_iter=int(train.get("max_iter", 1000)),
                class_weight="balanced",
            )
        )
        rlb.fit(X_train, y_train)

        best_k, _cv_scores = self.buscar_mejor_k(X, y)
        knn = make_pipeline(
            KNeighborsClassifier(
                n_neighbors=best_k,
                weights="distance",
            )
        )
        knn.fit(X_train, y_train)

        max_depth = train.get("max_depth")
        arbol = DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state,
            max_depth=max_depth,
            class_weight="balanced",
        )
        arbol.fit(X_train, y_train)

        rf = RandomForestClassifier(
            n_estimators=int(train["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(train.get("min_samples_leaf", 3)),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        return {
            "Regresión Logística": rlb,
            "KNN": knn,
            "Árbol de Decisión": arbol,
            "Random Forest": rf,
        }, rf

    def guardar_modelo(self, modelo, nombre_archivo="best_model.joblib"):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ruta = MODELS_DIR / nombre_archivo
        joblib.dump(modelo, ruta)
        print(f"[ModelTrainer] Modelo guardado en: {ruta}")
        return ruta

    def train(self, X_train, y_train) -> None:
        X = pd.concat([X_train], axis=0)
        y = pd.concat([y_train], axis=0)
        modelos, _rf = self._entrenar_modelos(X_train, y_train, X, y)
        self.model = modelos["Random Forest"]

    def evaluate(self, X_test, y_test) -> None:
        if self.model is None:
            raise RuntimeError("El modelo no está entrenado.")
        resultado = self.evaluator.evaluar_modelo(
            "Random Forest", self.model, X_test, y_test
        )
        self.evaluator.save_dvc_metrics(self.model, X_test, y_test, resultado)

    def run(self) -> None:
        print("Cargando dataset...")
        df = self.cargar_dataset()
        X, y = self.separar_variables(df)
        print(f"Dataset: {df.shape} | predictoras: {X.shape}")

        X_train, X_test, y_train, y_test = self.prepare_data()
        modelos, rf = self._entrenar_modelos(X_train, y_train, X, y)

        resultados_df = self.evaluator.evaluar_modelos(modelos, X_test, y_test)
        print("\nResultados:")
        print(
            resultados_df[
                [
                    "Modelo",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                    "avg_precision",
                ]
            ]
        )

        selection_metric = self.params["train"].get("selection_metric", "F1")
        best_model_name = resultados_df.loc[
            resultados_df[selection_metric].idxmax(),
            "Modelo",
        ]
        self.model = modelos[best_model_name]
        print(f"\nMejor modelo seleccionado: {best_model_name}")

        self.guardar_modelo(self.model)
        self.evaluator.save_dvc_metrics(
            self.model,
            X_test,
            y_test,
            resultados_df[resultados_df["Modelo"] == best_model_name].iloc[0],
        )

        graficar_roc_mejor_modelo(self.model, best_model_name, X_test, y_test)
        graficar_roc_modelos(modelos, X_test, y_test)
        graficar_comparacion_metricas(resultados_df)

        importance_df = calcular_permutation_importance(self.model, X_test, y_test)
        graficar_permutation_importance(importance_df, best_model_name)
        graficar_importancia_random_forest(rf, X.columns)

        self.predictor.model = self.model
        _scoring_df, decile_table = self.predictor.decile_table(X_test, y_test)
        graficar_ganancia_acumulada(decile_table, best_model_name)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        resultados_df.drop(columns=["Matriz_Confusion"], errors="ignore").to_csv(
            REPORTS_DIR / "metricas_modelos.csv",
            index=False,
        )
        importance_df.to_csv(REPORTS_DIR / "importancia_variables.csv", index=False)
        decile_table.to_csv(REPORTS_DIR / "tabla_deciles.csv", index=False)

        self._log_to_mlflow(
            modelos=modelos,
            resultados_df=resultados_df,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            best_model_name=best_model_name,
        )

    def _log_to_mlflow(
        self,
        modelos,
        resultados_df,
        X_train,
        y_train,
        X_test,
        y_test,
        best_model_name,
    ) -> None:
        mlflow_cfg = self.params.get("mlflow") or {}
        if not mlflow_cfg.get("enabled", False):
            return

        from caso_berka_model.mlflow_engine.run import log_and_register

        log_and_register(
            modelos=modelos,
            resultados_df=resultados_df,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            best_model_name=best_model_name,
            params=self.params,
        )


def main():
    ModelTrainer().run()


if __name__ == "__main__":
    main()
