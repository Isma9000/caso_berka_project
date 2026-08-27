from __future__ import annotations

import os
import tempfile
from typing import Any

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

from caso_berka_model.mlflow_engine.artifacts import ArtifactGenerator
from caso_berka_model.mlflow_engine.evaluator import AutomatedEvaluator
from caso_berka_model.mlflow_engine.pyfunc import EnterpriseDecisionWrapper
from caso_berka_model.mlflow_engine.settings import resolve_tracking_uri, sanitize_params


class MLflowEnterpriseTrainer:
    """Tracking MLflow: run padre, nested runs por modelo, PyFunc y evaluate."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = resolve_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def _log_candidate_charts(self, model, X_test, y_test) -> None:
        y_pred = model.predict(X_test)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = ArtifactGenerator.plot_confusion_matrix(
                y_test,
                y_pred,
                output_path=os.path.join(tmp_dir, "confusion_matrix.png"),
            )
            roc_path = ArtifactGenerator.plot_roc_curve(
                model,
                X_test,
                y_test,
                output_path=os.path.join(tmp_dir, "roc_curve.png"),
            )
            mlflow.log_artifact(cm_path, artifact_path="charts")
            mlflow.log_artifact(roc_path, artifact_path="charts")

    def _log_metrics_from_row(self, row: pd.Series) -> None:
        mlflow.log_metric("accuracy", float(row["Accuracy"]))
        mlflow.log_metric("precision", float(row["Precision"]))
        mlflow.log_metric("recall", float(row["Recall"]))
        mlflow.log_metric("f1_score", float(row["F1"]))
        mlflow.log_metric("roc_auc", float(row["ROC_AUC"]))
        if "avg_precision" in row:
            mlflow.log_metric("avg_precision", float(row["avg_precision"]))

    def _log_sklearn(self, model, artifact_name: str, **kwargs):
        pip_requirements = kwargs.pop(
            "pip_requirements",
            ["scikit-learn", "pandas", "numpy"],
        )
        try:
            return mlflow.sklearn.log_model(
                sk_model=model,
                name=artifact_name,
                pip_requirements=pip_requirements,
                **kwargs,
            )
        except TypeError:
            return mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_name,
                pip_requirements=pip_requirements,
                **kwargs,
            )

    def _log_pyfunc(self, python_model, artifact_name: str, **kwargs):
        pip_requirements = kwargs.pop(
            "pip_requirements",
            ["scikit-learn", "pandas", "numpy", "mlflow"],
        )
        try:
            return mlflow.pyfunc.log_model(
                python_model=python_model,
                name=artifact_name,
                pip_requirements=pip_requirements,
                **kwargs,
            )
        except TypeError:
            return mlflow.pyfunc.log_model(
                python_model=python_model,
                artifact_path=artifact_name,
                pip_requirements=pip_requirements,
                **kwargs,
            )

    def log_trained_models(
        self,
        modelos: dict[str, Any],
        resultados_df: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train,
        X_test: pd.DataFrame,
        y_test,
        best_model_name: str,
        params: dict[str, Any],
        lineage_tags: dict[str, str] | None = None,
        register_model_name: str | None = None,
        decision_threshold: float = 0.65,
        high_confidence: float = 0.85,
        run_evaluate: bool = True,
        run_name: str = "Berka_Model_Comparison",
        target_col: str = "buen_cliente",
    ) -> str:
        best_model = modelos[best_model_name]
        best_row = resultados_df.loc[
            resultados_df["Modelo"] == best_model_name
        ].iloc[0]
        tags = {
            "environment": "Production_Candidate",
            "architecture": "POO_Modular",
            "owner": "Team_8",
            "project": "Caso_Berka",
            "framework": "scikit-learn",
            "best_model": best_model_name,
            **(lineage_tags or {}),
        }

        with mlflow.start_run(run_name=run_name) as parent_run:
            mlflow.set_tags(tags)
            mlflow.log_params(sanitize_params(params))
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("decision_threshold", decision_threshold)
            self._log_metrics_from_row(best_row)

            for nombre, modelo in modelos.items():
                row = resultados_df.loc[resultados_df["Modelo"] == nombre].iloc[0]
                with mlflow.start_run(run_name=nombre, nested=True):
                    mlflow.set_tag("model_family", nombre)
                    mlflow.set_tag("is_best_model", str(nombre == best_model_name))
                    self._log_metrics_from_row(row)
                    self._log_candidate_charts(modelo, X_test, y_test)

            input_example = X_test.iloc[:5]
            signature = infer_signature(X_train, best_model.predict(X_train))
            model_info = self._log_sklearn(
                best_model,
                "sklearn_base_model",
                signature=signature,
                input_example=input_example,
            )

            wrapper = EnterpriseDecisionWrapper(
                trained_model=best_model,
                decision_threshold=decision_threshold,
                high_confidence=high_confidence,
            )
            pyfunc_example = wrapper.predict(None, input_example)
            pyfunc_signature = infer_signature(input_example, pyfunc_example)
            self._log_pyfunc(
                wrapper,
                "custom_decision_model",
                signature=pyfunc_signature,
                input_example=input_example,
                registered_model_name=register_model_name,
            )

            if run_evaluate:
                try:
                    AutomatedEvaluator.evaluate_model(
                        model_uri=model_info.model_uri,
                        X_test=X_test,
                        y_test=y_test,
                        target_col=target_col,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[MLflow] mlflow.evaluate omitido: {exc}")

            print(
                f"[MLflow] Run padre={parent_run.info.run_id} | "
                f"mejor={best_model_name} | F1={float(best_row['F1']):.4f}"
            )
            return parent_run.info.run_id
