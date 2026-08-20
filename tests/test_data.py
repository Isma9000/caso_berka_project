import pandas as pd

from caso_berka_model.dataset import DataProcessor
from caso_berka_model.features import FeatureEngineer
from caso_berka_model.modeling.predict import Predictor
from caso_berka_model.modeling.train import Evaluator, ModelTrainer


def test_feature_engineer_objetivo():
    engineer = FeatureEngineer()
    df = pd.DataFrame(
        {
            "client_id": [1, 2],
            "total_ingresos": [100.0, 10.0],
            "total_transacciones": [20, 2],
            "saldo_promedio": [50.0, 5.0],
            "moroso": [0, 1],
        }
    )
    resultado = engineer.crear_variable_objetivo(df)
    assert "buen_cliente" in resultado.columns
    assert set(resultado["buen_cliente"].unique()).issubset({0, 1})


def test_clases_pipeline_existen():
    assert DataProcessor is not None
    assert FeatureEngineer is not None
    assert ModelTrainer is not None
    assert Evaluator is not None
    assert Predictor is not None
