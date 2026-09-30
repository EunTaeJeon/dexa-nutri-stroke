from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model import build_estimator

__all__ = ["SelectionResult", "shap_rfe"]


@dataclass
class SelectionResult:
    selected: list[str]
    score: float
    history: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def _fold_shap_importance(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import shap

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    importance = np.zeros(X.shape[1], dtype=float)
    oof = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]

        transform = est[:-1]
        classifier = est[-1]
        X_held = transform.transform(X.iloc[test_idx])

        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(X_held)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]

        contrib = np.abs(values).mean(axis=0)
        index = {c: i for i, c in enumerate(X.columns)}
        for col, val in zip(list(X_held.columns), contrib):
            importance[index[col]] += val

    return importance / n_folds, oof


def shap_rfe(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: tuple[str, ...] = (),
    params: dict | None = None,
    *,
    min_features: int = 5,
    step: float = 0.1,
    n_folds: int = 10,
    seed: int = 0,
    missing_threshold: float = 0.20,
    contamination: float = 0.05,
    verbose: bool = True,
) -> SelectionResult:
    if not 0.0 < step < 1.0:
        raise ValueError("step must lie in (0, 1)")

    features = list(X.columns)
    best_subset, best_score = list(features), -np.inf
    history: list[dict] = []

    while len(features) >= min_features:
        cats = tuple(c for c in categorical if c in features)
        estimator = build_estimator(
            cats,
            params,
            random_state=seed,
            missing_threshold=missing_threshold,
            contamination=contamination,
        )
        importance, oof = _fold_shap_importance(
            estimator, X[features], y, n_folds=n_folds, seed=seed
        )
        score = roc_auc_score(y, oof)
        history.append({"n_features": len(features), "auroc": score})
        if verbose:
            print(f"[rfe] {len(features):3d} features -> CV AUROC {score:.4f}")

        if score > best_score:
            best_score, best_subset = score, list(features)

        n_drop = max(1, int(round(step * len(features))))
        if len(features) - n_drop < min_features:
            break
        order = np.argsort(importance)
        drop = {features[i] for i in order[:n_drop]}
        features = [f for f in features if f not in drop]

    if verbose:
        print(f"[rfe] selected {len(best_subset)} features, CV AUROC {best_score:.4f}")
    return SelectionResult(selected=best_subset, score=best_score, history=history)
