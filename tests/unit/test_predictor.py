import pytest
from sklearn.ensemble import RandomForestClassifier

from caso_berka_model.modeling.predict import Predictor

pytestmark = pytest.mark.unit


def test_decile_table_returns_ten_deciles(synthetic_tabla_minable):
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

    model = RandomForestClassifier(n_estimators=8, random_state=42)
    model.fit(X, y)

    predictor = Predictor()
    predictor.model = model
    _scoring_df, decile_table = predictor.decile_table(X, y)

    assert len(decile_table) == 10
    assert "decil_riesgo" in decile_table.columns
    assert "captura_buen_cliente_acum" in decile_table.columns
    assert decile_table["decil_riesgo"].tolist() == list(range(1, 11))
