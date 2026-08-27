import json
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from caso_berka_model.modeling.train import ModelTrainer

pytestmark = pytest.mark.integration


def test_model_trainer_train_evaluate_and_save(
    synthetic_tabla_minable, tmp_path, tmp_params_yaml, monkeypatch
):
    data_path = tmp_path / "tabla_minable.csv"
    metrics_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    synthetic_tabla_minable.to_csv(data_path, index=False)

    monkeypatch.setattr(
        "caso_berka_model.modeling.train.MODELS_DIR",
        models_dir,
    )

    trainer = ModelTrainer(
        params_path=tmp_params_yaml,
        data_path=data_path,
        metrics_dir=metrics_dir,
    )
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    model_path = trainer.guardar_modelo(trainer.model)

    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert isinstance(loaded, RandomForestClassifier)

    eval_path = metrics_dir / "eval.json"
    assert eval_path.exists()
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    assert "f1_score" in metrics
