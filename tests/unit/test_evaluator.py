import json

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from caso_berka_model.modeling.train import Evaluator

pytestmark = pytest.mark.unit


def test_save_dvc_metrics_writes_eval_json(tmp_path, synthetic_tabla_minable):
    metrics_dir = tmp_path / "metrics"
    evaluator = Evaluator(metrics_dir)

    feature_cols = [
        "birth_number",
        "date",
        "cantidad_ingresos",
        "total_egresos",
        "cantidad_egresos",
        "tiene_prestamo",
        "monto_prestamo",
        "tiene_tarjeta",
    ]
    X = synthetic_tabla_minable[feature_cols]
    y = synthetic_tabla_minable["buen_cliente"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    resultado = evaluator.evaluar_modelo("Random Forest", model, X.iloc[:10], y.iloc[:10])
    evaluator.save_dvc_metrics(model, X.iloc[:10], y.iloc[:10], resultado)

    eval_path = metrics_dir / "eval.json"
    plots_path = metrics_dir / "plots.csv"

    assert eval_path.exists()
    assert plots_path.exists()

    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}

    plots_df = pd.read_csv(plots_path)
    assert {"actual", "predicted"}.issubset(plots_df.columns)
