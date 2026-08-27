from __future__ import annotations

import mlflow
import pandas as pd


class AutomatedEvaluator:
    """Evaluación nativa de MLflow sobre el modelo sklearn base."""

    @staticmethod
    def evaluate_model(
        model_uri: str,
        X_test: pd.DataFrame,
        y_test,
        target_col: str = "buen_cliente",
        dataset_name: str = "test_evaluation_dataset",
    ):
        eval_data = X_test.copy().reset_index(drop=True)
        eval_data[target_col] = pd.Series(y_test).reset_index(drop=True)
        payload = {
            "model": model_uri,
            "data": eval_data,
            "targets": target_col,
            "model_type": "classifier",
        }

        models_evaluate = getattr(mlflow, "models", None)
        if models_evaluate is not None and hasattr(models_evaluate, "evaluate"):
            try:
                return models_evaluate.evaluate(**payload)
            except TypeError:
                pass

        try:
            return mlflow.evaluate(**payload, dataset_name=dataset_name)
        except TypeError:
            return mlflow.evaluate(**payload)
