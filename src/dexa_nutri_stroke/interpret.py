from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["shap_values_for", "global_importance", "dependence", "inflection_point"]


def _unwrap(model):
    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(model, CalibratedClassifierCV):
        inner = model.calibrated_classifiers_[0].estimator
        return inner[:-1], inner[-1]
    return model[:-1], model[-1]


def shap_values_for(model, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    import shap

    transform, classifier = _unwrap(model)
    X_t = transform.transform(X)

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values, X_t


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    values, X_t = shap_values_for(model, X)
    imp = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(X_t.columns), "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class Dependence:
    feature: str
    x: np.ndarray
    shap: np.ndarray
    curve_x: np.ndarray
    curve_y: np.ndarray


def dependence(model, X: pd.DataFrame, feature: str, *, frac: float = 0.5) -> Dependence:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    values, X_t = shap_values_for(model, X)
    if feature not in X_t.columns:
        raise KeyError(f"{feature!r} is not present after preprocessing")
    j = list(X_t.columns).index(feature)

    x = X_t[feature].to_numpy(dtype=float)
    s = values[:, j]
    order = np.argsort(x)
    smoothed = lowess(s[order], x[order], frac=frac, return_sorted=False)
    return Dependence(feature, x, s, x[order], smoothed)


def _crossing(curve_x: np.ndarray, curve_y: np.ndarray) -> float:
    sign = np.sign(curve_y)
    changes = np.where(np.diff(sign) != 0)[0]
    if len(changes) == 0:
        return np.nan
    i = changes[0]
    y0, y1 = curve_y[i], curve_y[i + 1]
    x0, x1 = curve_x[i], curve_x[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))


def inflection_point(
    model,
    X: pd.DataFrame,
    feature: str,
    *,
    n_boot: int = 200,
    seed: int = 0,
    frac: float = 0.5,
) -> dict:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    dep = dependence(model, X, feature, frac=frac)
    point = _crossing(dep.curve_x, dep.curve_y)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(dep.x))
    draws = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        xb, sb = dep.x[take], dep.shap[take]
        order = np.argsort(xb)
        try:
            yb = lowess(sb[order], xb[order], frac=frac, return_sorted=False)
        except Exception:
            continue
        val = _crossing(xb[order], yb)
        if np.isfinite(val):
            draws.append(val)

    if len(draws) < 20:
        return {"feature": feature, "inflection": point, "lower": np.nan, "upper": np.nan}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "feature": feature,
        "inflection": point,
        "lower": float(lo),
        "upper": float(hi),
    }
