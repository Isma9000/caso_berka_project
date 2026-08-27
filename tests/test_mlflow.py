import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from caso_berka_model.mlflow_engine.lineage import DataLineage
from caso_berka_model.mlflow_engine.pyfunc import EnterpriseDecisionWrapper
from caso_berka_model.mlflow_engine.registry import MLflowGovernanceManager
from caso_berka_model.mlflow_engine.trainer import MLflowEnterpriseTrainer


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


def test_mlflow_logs_metrics_and_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    X_train, X_test, y_train, y_test = _toy_dataset()
    model = RandomForestClassifier(n_estimators=8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    resultados = pd.DataFrame(
        [
            {
                "Modelo": "Random Forest",
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1": f1_score(y_test, preds, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, proba),
                "avg_precision": average_precision_score(y_test, proba),
            }
        ]
    )

    trainer = MLflowEnterpriseTrainer(
        experiment_name="Test_Berka_MLflow",
        tracking_uri=tracking_uri,
    )
    run_id = trainer.log_trained_models(
        modelos={"Random Forest": model},
        resultados_df=resultados,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        best_model_name="Random Forest",
        params={"n_estimators": 8, "max_depth": None},
        lineage_tags={"git_commit": "test", "dvc_status": "synced"},
        register_model_name="Test_Berka_Model",
        decision_threshold=0.5,
        run_evaluate=False,
        run_name="test_run",
    )
    assert run_id
    assert (tmp_path / "mlflow.db").exists()

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=["Test_Berka_MLflow"])
    assert not runs.empty
    assert "metrics.accuracy" in runs.columns
    assert "metrics.f1_score" in runs.columns

    governance = MLflowGovernanceManager(tracking_uri=tracking_uri)
    version = governance.latest_version("Test_Berka_Model")
    governance.promote_to_production("Test_Berka_Model", version)
    loaded = governance.load_latest_production_model("Test_Berka_Model")
    predictions = loaded.predict(X_test.iloc[:2])
    assert "prediction" in predictions.columns
    assert "probability" in predictions.columns
    assert len(predictions) == 2
