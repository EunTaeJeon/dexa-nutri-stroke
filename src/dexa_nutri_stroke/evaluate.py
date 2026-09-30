from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

__all__ = [
    "Discrimination",
    "Calibration",
    "auroc_ci",
    "delong_test",
    "continuous_nri",
    "calibration_metrics",
    "operating_point",
    "evaluate_model",
    "subgroup_table",
]


@dataclass
class Discrimination:
    auroc: float
    lower: float
    upper: float
    n: int
    events: int

    def as_dict(self) -> dict:
        return asdict(self)


def auroc_ci(
    y: np.ndarray, p: np.ndarray, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Discrimination:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    point = roc_auc_score(y, p)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    draws = np.empty(n_boot, dtype=float)
    kept = 0
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        draws[kept] = roc_auc_score(y[take], p[take])
        kept += 1
    lo, hi = np.percentile(draws[:kept], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Discrimination(point, float(lo), float(hi), len(y), int(y.sum()))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(predictions: np.ndarray, y: np.ndarray):
    pos = predictions[:, y == 1]
    neg = predictions[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        raise ValueError("both outcome classes must be present")
    k = predictions.shape[0]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    aucs, cov = _delong_components(np.vstack([np.asarray(p1), np.asarray(p2)]), y)
    contrast = np.array([[1.0, -1.0]])
    var = (contrast @ cov @ contrast.T).item()
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auroc_1": aucs[0], "auroc_2": aucs[1], "difference": diff, "p_value": np.nan}
    z = diff / np.sqrt(var)
    return {
        "auroc_1": float(aucs[0]),
        "auroc_2": float(aucs[1]),
        "difference": diff,
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def continuous_nri(
    y: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    y = np.asarray(y, dtype=int)
    p_old = np.asarray(p_old, dtype=float)
    p_new = np.asarray(p_new, dtype=float)

    def statistic(yy, po, pn):
        up = pn > po
        down = pn < po
        ev, ne = yy == 1, yy == 0
        if ev.sum() == 0 or ne.sum() == 0:
            return np.nan
        return (up[ev].mean() - down[ev].mean()) + (down[ne].mean() - up[ne].mean())

    point = statistic(y, p_old, p_new)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    draws = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        val = statistic(y[take], p_old[take], p_new[take])
        if np.isfinite(val):
            draws.append(val)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"nri": float(point), "lower": float(lo), "upper": float(hi)}


@dataclass
class Calibration:
    brier: float
    slope: float
    intercept: float
    ici: float
    e50: float
    e90: float

    def as_dict(self) -> dict:
        return asdict(self)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Calibration:
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    lp = _logit(p)

    try:
        slope = float(
            sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        )
    except Exception:
        slope = np.nan
    try:
        intercept = float(
            sm.GLM(
                y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp
            ).fit().params[0]
        )
    except Exception:
        intercept = np.nan

    order = np.argsort(p)
    smoothed = lowess(y[order], p[order], frac=0.6, return_sorted=False)
    gap = np.abs(smoothed - p[order])
    return Calibration(
        brier=float(brier_score_loss(y, p)),
        slope=slope,
        intercept=intercept,
        ici=float(gap.mean()),
        e50=float(np.percentile(gap, 50)),
        e90=float(np.percentile(gap, 90)),
    )


def _confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a: int, b: int) -> float:
        return float(a / b) if b else np.nan

    return {
        "threshold": float(threshold),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, len(y)),
    }


def operating_point(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    best = int(np.argmax(tpr - fpr))
    return _confusion_metrics(y, p, float(thresholds[best]))


def evaluate_model(
    y: np.ndarray, p: np.ndarray, *, threshold: float | None = None, seed: int = 0
) -> dict:
    disc = auroc_ci(y, p, seed=seed)
    cal = calibration_metrics(y, p)
    point = operating_point(y, p) if threshold is None else _confusion_metrics(y, p, threshold)
    return {**disc.as_dict(), **cal.as_dict(), **point}


def subgroup_table(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_new: np.ndarray,
    subgroups: dict[str, np.ndarray],
    *,
    min_size: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests

    rows = []
    for name, mask in subgroups.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < min_size or len(np.unique(y[mask])) < 2:
            continue
        ref = auroc_ci(y[mask], p_ref[mask], n_boot=1000, seed=seed)
        new = auroc_ci(y[mask], p_new[mask], n_boot=1000, seed=seed)
        test = delong_test(y[mask], p_new[mask], p_ref[mask])
        rows.append(
            {
                "subgroup": name,
                "n": int(mask.sum()),
                "events": int(y[mask].sum()),
                "auroc_ref": ref.auroc,
                "auroc_ref_lower": ref.lower,
                "auroc_ref_upper": ref.upper,
                "auroc_new": new.auroc,
                "auroc_new_lower": new.lower,
                "auroc_new_upper": new.upper,
                "p_value": test["p_value"],
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        finite = table["p_value"].notna()
        table["q_value"] = np.nan
        if finite.any():
            table.loc[finite, "q_value"] = multipletests(
                table.loc[finite, "p_value"], method="fdr_bh"
            )[1]
    return table
