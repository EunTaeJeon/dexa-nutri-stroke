from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor

__all__ = ["SearchSpace", "build_estimator", "tune", "fit_calibrated", "cross_val_probabilities"]

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_LGBM = False


@dataclass(frozen=True)
class SearchSpace:
    num_leaves: tuple[int, int] = (8, 64)
    max_depth: tuple[int, int] = (2, 7)
    learning_rate: tuple[float, float] = (0.01, 0.2)
    min_child_samples: tuple[int, int] = (10, 60)
    subsample: tuple[float, float] = (0.6, 1.0)
    colsample_bytree: tuple[float, float] = (0.4, 1.0)
    reg_lambda: tuple[float, float] = (1e-3, 10.0)


def build_estimator(
    categorical: tuple[str, ...] = (),
    params: dict | None = None,
    *,
    random_state: int = 0,
    missing_threshold: float = 0.20,
    contamination: float = 0.05,
) -> Pipeline:
    params = dict(params or {})
    if _HAS_LGBM:
        defaults = dict(
            objective="binary",
            n_estimators=400,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        defaults.update(params)
        clf = LGBMClassifier(**defaults)
    else:
        allowed = {"learning_rate", "max_depth"}
        clf = HistGradientBoostingClassifier(
            random_state=random_state,
            **{k: v for k, v in params.items() if k in allowed},
        )

    steps = build_preprocessor(
        categorical,
        missing_threshold=missing_threshold,
        contamination=contamination,
        random_state=random_state,
    )
    steps.append(("classifier", clf))
    return Pipeline(steps)


def _suggest(trial, space: SearchSpace) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", *space.num_leaves),
        "max_depth": trial.suggest_int("max_depth", *space.max_depth),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *space.min_child_samples),
        "subsample": trial.suggest_float("subsample", *space.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda, log=True),
    }


def cross_val_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_folds: int = 10,
    seed: int = 0,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return cross_val_predict(
        clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=None
    )[:, 1]


def tune(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: tuple[str, ...] = (),
    *,
    space: SearchSpace | None = None,
    n_trials: int = 50,
    n_folds: int = 10,
    seed: int = 0,
    missing_threshold: float = 0.20,
    contamination: float = 0.05,
) -> dict:
    space = space or SearchSpace()
    try:
        import optuna
    except ImportError:
        print("[model] optuna not installed; using default hyperparameters")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = _suggest(trial, space)
        est = build_estimator(
            categorical,
            params,
            random_state=seed,
            missing_threshold=missing_threshold,
            contamination=contamination,
        )
        proba = cross_val_probabilities(est, X, y, n_folds=n_folds, seed=seed)
        return roc_auc_score(y, proba)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[model] best cross-validated AUROC {study.best_value:.4f} after {n_trials} trials")
    return dict(study.best_params)


def fit_calibrated(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: tuple[str, ...] = (),
    params: dict | None = None,
    *,
    n_folds: int = 10,
    seed: int = 0,
    missing_threshold: float = 0.20,
    contamination: float = 0.05,
) -> ClassifierMixin:
    base = build_estimator(
        categorical,
        params,
        random_state=seed,
        missing_threshold=missing_threshold,
        contamination=contamination,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    model.fit(X, y)
    return model
