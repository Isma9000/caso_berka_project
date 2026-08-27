import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from caso_berka_model.mlflow_engine.lineage import DataLineage
from caso_berka_model.mlflow_engine.pyfunc import EnterpriseDecisionWrapper

pytestmark = pytest.mark.unit


def _toy_dataset():
    features, target = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X = pd.DataFrame(features, columns=[f"f{i}" for i in range(4)])
    y = pd.Series(target, name="buen_cliente")
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def test_lineage_metadata():
    metadata = DataLineage().get_lineage_metadata()
    assert "git_commit" in metadata
    assert "dvc_status" in metadata
    assert metadata["git_commit"]


def test_pyfunc_wrapper_predict():
    X_train, X_test, y_train, _y_test = _toy_dataset()
    model = RandomForestClassifier(n_estimators=8, random_state=42)
    model.fit(X_train, y_train)
    wrapper = EnterpriseDecisionWrapper(model, decision_threshold=0.5)
    output = wrapper.predict(None, X_test.iloc[:3])
    assert list(output.columns) == [
        "probability",
        "prediction",
        "high_confidence_flag",
    ]
    assert len(output) == 3


def test_pyfunc_wrapper_uses_custom_threshold():
    class _FixedProbaModel:
        def predict_proba(self, X):
            import numpy as np

            n = len(X)
            return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])

    wrapper = EnterpriseDecisionWrapper(
        _FixedProbaModel(),
        decision_threshold=0.99,
        high_confidence=0.995,
    )
    output = wrapper.predict(None, pd.DataFrame({"f0": [0, 1, 2]}))
    assert (output["prediction"] == 0).all()
    assert (output["high_confidence_flag"] == 0).all()
