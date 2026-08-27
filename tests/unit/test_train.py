import pytest

from caso_berka_model.modeling.train import ModelTrainer

pytestmark = pytest.mark.unit


def test_prepare_data_uses_prepare_split_ratio(
    synthetic_tabla_minable, tmp_path, tmp_params_yaml
):
    data_path = tmp_path / "tabla_minable.csv"
    synthetic_tabla_minable.to_csv(data_path, index=False)

    trainer = ModelTrainer(
        params_path=tmp_params_yaml,
        data_path=data_path,
        metrics_dir=tmp_path / "metrics",
    )
    X_train, X_test, y_train, y_test = trainer.prepare_data()

    assert len(X_train) + len(X_test) == len(synthetic_tabla_minable)
    assert len(X_test) == pytest.approx(
        len(synthetic_tabla_minable) * 0.30, abs=2
    )
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_separar_variables_reads_target_from_params(
    synthetic_tabla_minable, tmp_params_yaml
):
    trainer = ModelTrainer(params_path=tmp_params_yaml)
    X, y = trainer.separar_variables(synthetic_tabla_minable)

    assert trainer.params["train"]["target_col"] == "buen_cliente"
    assert "buen_cliente" not in X.columns
    assert len(y) == len(synthetic_tabla_minable)
