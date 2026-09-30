from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .data import Dataset, Schema, load_dataset, split_once
from .evaluate import (
    continuous_nri,
    delong_test,
    evaluate_model,
    operating_point,
    subgroup_table,
)
from .features import shap_rfe
from .model import build_estimator, cross_val_probabilities, fit_calibrated, tune

__all__ = ["ModelResult", "ExperimentResult", "run_experiment", "save_results"]


@dataclass
class ModelResult:
    name: str
    features: list[str]
    params: dict
    threshold: float
    cv_auroc: float
    valid_probabilities: np.ndarray
    metrics: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    dataset: Dataset
    models: dict[str, ModelResult]
    comparison: dict = field(default_factory=dict)
    subgroups: pd.DataFrame | None = None

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, m in self.models.items():
            rows.append({"model": name, "n_features": len(m.features), **m.metrics})
        return pd.DataFrame(rows)


def _fit_one(name: str, ds: Dataset, model_key: str, cfg: ExperimentConfig) -> ModelResult:
    X_tr, y_tr = ds.xy("train", model_key)
    categorical = tuple(c for c in ds.schema.categorical if c in X_tr.columns)

    print(f"\n[{name}] {X_tr.shape[1]} candidate features, {len(y_tr)} training rows")

    if cfg.select_features:
        selection = shap_rfe(
            X_tr,
            y_tr,
            categorical,
            min_features=cfg.min_features,
            step=cfg.rfe_step,
            n_folds=cfg.n_folds,
            seed=cfg.seed,
            missing_threshold=cfg.missing_threshold,
            contamination=cfg.contamination,
        )
        features = selection.selected
    else:
        features = list(X_tr.columns)

    X_tr = X_tr[features]
    categorical = tuple(c for c in categorical if c in features)

    params = tune(
        X_tr,
        y_tr,
        categorical,
        n_trials=cfg.n_trials,
        n_folds=cfg.n_folds,
        seed=cfg.seed,
        missing_threshold=cfg.missing_threshold,
        contamination=cfg.contamination,
    )

    estimator = build_estimator(
        categorical,
        params,
        random_state=cfg.seed,
        missing_threshold=cfg.missing_threshold,
        contamination=cfg.contamination,
    )
    oof = cross_val_probabilities(estimator, X_tr, y_tr, n_folds=cfg.n_folds, seed=cfg.seed)
    cv_auroc = float(roc_auc_score(y_tr, oof))
    threshold = float(operating_point(y_tr, oof)["threshold"])
    print(f"[{name}] cross-validated AUROC {cv_auroc:.4f}, threshold {threshold:.4f}")

    model = fit_calibrated(
        X_tr,
        y_tr,
        categorical,
        params,
        n_folds=cfg.n_folds,
        seed=cfg.seed,
        missing_threshold=cfg.missing_threshold,
        contamination=cfg.contamination,
    )

    X_va, y_va = ds.xy("valid", model_key)
    p_va = model.predict_proba(X_va[features])[:, 1]
    metrics = evaluate_model(y_va, p_va, threshold=threshold, seed=cfg.seed)
    metrics["cv_auroc"] = cv_auroc
    print(
        f"[{name}] validation AUROC {metrics['auroc']:.3f} "
        f"[{metrics['lower']:.3f}-{metrics['upper']:.3f}], Brier {metrics['brier']:.3f}"
    )

    return ModelResult(
        name=name,
        features=features,
        params=params,
        threshold=threshold,
        cv_auroc=cv_auroc,
        valid_probabilities=p_va,
        metrics=metrics,
    )


def run_experiment(
    data_path: str | Path,
    schema: Schema,
    cfg: ExperimentConfig,
    *,
    subgroups: dict[str, np.ndarray] | None = None,
) -> ExperimentResult:
    df = load_dataset(data_path, schema)
    ds = split_once(df, schema, valid_size=cfg.valid_size, seed=cfg.seed)

    results = {
        "Clin": _fit_one("Clin", ds, "clin", cfg),
        "Clin-DEXA-Nutri": _fit_one("Clin-DEXA-Nutri", ds, "clin_dexa_nutri", cfg),
    }

    y_va = ds.valid[schema.outcome].to_numpy(dtype=int)
    p_ref = results["Clin"].valid_probabilities
    p_new = results["Clin-DEXA-Nutri"].valid_probabilities

    comparison = {
        "delong": delong_test(y_va, p_new, p_ref),
        "nri": continuous_nri(y_va, p_ref, p_new, seed=cfg.seed),
    }
    print(
        f"\n[compare] AUROC {comparison['delong']['auroc_1']:.3f} vs "
        f"{comparison['delong']['auroc_2']:.3f}, "
        f"difference {comparison['delong']['difference']:+.3f}, "
        f"P = {comparison['delong']['p_value']:.4f}"
    )

    table = None
    if subgroups:
        table = subgroup_table(y_va, p_ref, p_new, subgroups, seed=cfg.seed)

    result = ExperimentResult(dataset=ds, models=results, comparison=comparison, subgroups=table)

    if cfg.output_dir:
        save_results(result, cfg)
    return result


def save_results(result: ExperimentResult, cfg: ExperimentConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.summary().to_csv(out / "model_performance.csv", index=False)
    with open(out / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(result.comparison, f, indent=2)
    with open(out / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({k: v.features for k, v in result.models.items()}, f, indent=2)
    with open(out / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump({k: v.params for k, v in result.models.items()}, f, indent=2)
    if result.subgroups is not None and not result.subgroups.empty:
        result.subgroups.to_csv(out / "subgroup_performance.csv", index=False)
    print(f"\n[io] results written to {out.resolve()}")
