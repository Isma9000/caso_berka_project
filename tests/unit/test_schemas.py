import pytest
from pydantic import ValidationError

from caso_berka_model.api.schemas import BerkaFeatures, PredictionRequest

pytestmark = pytest.mark.unit


def test_prediction_request_requires_all_features():
    payload = {
        "birth_number": 1.0,
        "date": 2.0,
        "cantidad_ingresos": 0.5,
        "total_egresos": -0.1,
        "cantidad_egresos": 1.0,
        "tiene_prestamo": 0.0,
        "monto_prestamo": -0.2,
        "tiene_tarjeta": 1.0,
    }
    request = PredictionRequest(data=[BerkaFeatures(**payload)])
    assert len(request.data) == 1


def test_prediction_request_rejects_missing_feature():
    with pytest.raises(ValidationError):
        BerkaFeatures(
            birth_number=1.0,
            date=2.0,
            cantidad_ingresos=0.5,
            total_egresos=-0.1,
            cantidad_egresos=1.0,
            tiene_prestamo=0.0,
            monto_prestamo=-0.2,
        )
