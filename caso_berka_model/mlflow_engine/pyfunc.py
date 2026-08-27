from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd


class EnterpriseDecisionWrapper(mlflow.pyfunc.PythonModel):
    """PyFunc con umbral de decisión y bandera de alta confianza para buen_cliente."""

    def __init__(
        self,
        trained_model: Any = None,
        decision_threshold: float = 0.65,
        high_confidence: float = 0.85,
    ):
        self.trained_model = trained_model
        self.decision_threshold = decision_threshold
        self.high_confidence = high_confidence

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            features = model_input
        else:
            features = pd.DataFrame(model_input)

        probabilities = self.trained_model.predict_proba(features)[:, 1]
        decisions = (probabilities >= self.decision_threshold).astype(int)
        confidence = (probabilities >= self.high_confidence).astype(int)
        return pd.DataFrame(
            {
                "probability": probabilities,
                "prediction": decisions,
                "high_confidence_flag": confidence,
            }
        )
