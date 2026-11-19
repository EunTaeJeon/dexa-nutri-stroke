from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["shap_values_for", "global_importance", "dependence", "inflection_point"]


def shap_values_for(
    model,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Explain model predictions and return the matching feature values.

    For a calibrated classifier, explain the event probability on the original
    predictors. The reference defaults to X. Pass development predictors for
    a baseline independent of the rows being explained.
    For an uncalibrated tree pipeline, return TreeSHAP on the raw tree output
    and the preprocessed predictors.
    """
    import shap
    from sklearn.calibration import CalibratedClassifierCV

    if background is not None:
        if (
            background.empty
            or not background.columns.is_unique
            or set(background.columns) != set(X.columns)
        ):
            raise ValueError("background must contain rows with the same feature columns as X")
        background = background.loc[:, X.columns]

    if isinstance(model, CalibratedClassifierCV):
        if X.empty or not X.columns.is_unique:
            raise ValueError("X must contain rows and uniquely named features")
        reference = X if background is None else background

        class ExactIndependent(shap.maskers.Independent):
            def invariants(self, row):
                # Numerically close values can fall on opposite sides of a tree split.
                return (self.data == row) | (np.isnan(self.data) & np.isnan(row))

        masker = ExactIndependent(reference.astype(float), max_samples=len(reference))

        def event_probability(rows):
            return model.predict_proba(rows)[:, 1]

        explainer = shap.PermutationExplainer(event_probability, masker, seed=seed)
        explanation = explainer(X.astype(float), max_evals="auto", silent=True)
        return explanation.values, X

    transform, classifier = model[:-1], model[-1]
    X_t = transform.transform(X)

    masker = None
    if background is not None:
        reference = transform.transform(background)
        masker = shap.maskers.Independent(reference, max_samples=len(reference))
    explainer = shap.TreeExplainer(classifier, data=masker)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values, X_t


def global_importance(
    model,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    values, X_t = shap_values_for(model, X, background=background, seed=seed)
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


def dependence(
    model,
    X: pd.DataFrame,
    feature: str,
    *,
    frac: float = 0.5,
    background: pd.DataFrame | None = None,
    seed: int = 0,
) -> Dependence:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    values, X_t = shap_values_for(model, X, background=background, seed=seed)
    if feature not in X_t.columns:
        raise KeyError(f"{feature!r} is not present in the explained features")
    j = list(X_t.columns).index(feature)

    x = X_t[feature].to_numpy(dtype=float, na_value=np.nan)
    s = values[:, j]
    observed = np.isfinite(x)
    if not observed.any():
        raise ValueError(f"{feature!r} has no observed values for the dependence curve")
    x, s = x[observed], s[observed]
    order = np.argsort(x)
    smoothed = lowess(s[order], x[order], frac=frac, return_sorted=False)
    return Dependence(feature, x, s, x[order], smoothed)


def _crossing(curve_x: np.ndarray, curve_y: np.ndarray) -> float:
    if not np.isfinite(curve_y).all():
        return np.nan
    nonzero = np.flatnonzero(curve_y)
    changes = np.flatnonzero(np.diff(np.sign(curve_y[nonzero])))
    if len(changes) != 1:
        return np.nan
    left, right = nonzero[changes[0]:changes[0] + 2]
    if right > left + 2:
        return np.nan
    if right == left + 2:
        return float(curve_x[left + 1])
    y0, y1 = curve_y[left], curve_y[right]
    x0, x1 = curve_x[left], curve_x[right]
    return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))


def inflection_point(
    model,
    X: pd.DataFrame,
    feature: str,
    *,
    n_boot: int = 200,
    seed: int = 0,
    frac: float = 0.5,
    background: pd.DataFrame | None = None,
) -> dict:
    """Estimate a single sign-changing zero of a fixed model's SHAP curve."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    dep = dependence(model, X, feature, frac=frac, background=background, seed=seed)
    point = _crossing(dep.curve_x, dep.curve_y)
    if not np.isfinite(point):
        return {"feature": feature, "inflection": point, "lower": np.nan, "upper": np.nan}

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
