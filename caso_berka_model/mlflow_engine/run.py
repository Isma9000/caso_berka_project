from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from caso_berka_model.config import PARAMS_PATH
from caso_berka_model.mlflow_engine.lineage import DataLineage
from caso_berka_model.mlflow_engine.registry import MLflowGovernanceManager
from caso_berka_model.mlflow_engine.settings import get_mlflow_config
from caso_berka_model.mlflow_engine.trainer import MLflowEnterpriseTrainer


def _training_params(params: dict[str, Any], best_model_name: str) -> dict[str, Any]:
    train = params.get("train") or {}
    prepare = params.get("prepare") or {}
    return {
        "n_estimators": train.get("n_estimators"),
        "max_depth": train.get("max_depth"),
        "min_samples_leaf": train.get("min_samples_leaf"),
        "max_iter": train.get("max_iter"),
        "knn_k_max": train.get("knn_k_max"),
        "selection_metric": train.get("selection_metric"),
        "split_ratio": prepare.get("split_ratio"),
        "random_state": prepare.get("random_state"),
        "target_col": train.get("target_col", "buen_cliente"),
        "best_model": best_model_name,
    }


def log_and_register(
    modelos: dict[str, Any],
    resultados_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    best_model_name: str,
    params: dict[str, Any],
) -> str | None:
    cfg = get_mlflow_config(params)
    if not cfg["enabled"]:
        print("[MLflow] Tracking desactivado en params.yaml")
        return None

    lineage = DataLineage().get_lineage_metadata()
    trainer = MLflowEnterpriseTrainer(
        experiment_name=cfg["experiment_name"],
        tracking_uri=cfg["tracking_uri"],
    )
    run_id = trainer.log_trained_models(
        modelos=modelos,
        resultados_df=resultados_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        best_model_name=best_model_name,
        params=_training_params(params, best_model_name),
        lineage_tags=lineage,
        register_model_name=cfg["registered_model_name"],
        decision_threshold=cfg["decision_threshold"],
        high_confidence=cfg["high_confidence"],
        run_evaluate=cfg["run_evaluate"],
        target_col=params.get("train", {}).get("target_col", "buen_cliente"),
    )

    if cfg["promote_to_production"] and cfg["registered_model_name"]:
        governance = MLflowGovernanceManager(tracking_uri=cfg["tracking_uri"])
        version = governance.latest_version(cfg["registered_model_name"])
        governance.promote_to_production(cfg["registered_model_name"], version)

    return run_id


def _load_params() -> dict[str, Any]:
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _demo_production_inference(params: dict[str, Any]) -> None:
    from caso_berka_model.modeling.train import ModelTrainer

    cfg = get_mlflow_config(params)
    if not cfg["enabled"] or not cfg["promote_to_production"]:
        return

    _, X_test, _, _y_test = ModelTrainer().prepare_data()
    governance = MLflowGovernanceManager(tracking_uri=cfg["tracking_uri"])
    production_model = governance.load_latest_production_model(
        cfg["registered_model_name"]
    )
    predictions = production_model.predict(X_test.iloc[:5])
    print("[MLflow] Inferencia Production (5 filas):")
    print(predictions)


def main() -> None:
    from caso_berka_model.modeling.train import ModelTrainer

    ModelTrainer().run()
    _demo_production_inference(_load_params())


if __name__ == "__main__":
    main()
