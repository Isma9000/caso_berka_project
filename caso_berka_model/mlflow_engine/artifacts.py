from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


class ArtifactGenerator:
    """Genera gráficos de diagnóstico para almacenar como artefactos en MLflow."""

    @staticmethod
    def plot_confusion_matrix(
        y_true,
        y_pred,
        output_path: str | Path = "confusion_matrix.png",
    ) -> str:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()
        return str(path)

    @staticmethod
    def plot_roc_curve(
        model,
        X_test,
        y_test,
        output_path: str | Path = "roc_curve.png",
    ) -> str:
        _fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Curva ROC")
        plt.tight_layout()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()
        return str(path)
