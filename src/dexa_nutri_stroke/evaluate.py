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
    "compare_at_thresholds",
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
    if min(m, n) < 2:
        raise ValueError("DeLong variance requires at least two observations per outcome class")
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


def _ratio(numerator, denominator):
    return np.divide(
        numerator, denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator != 0,
    )


def _confusion_metrics(tn, fp, fn, tp) -> dict:
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "diagnostic_or": _ratio(tp * tn, fp * fn),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
    }


def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(p, dtype=float) >= threshold
    counts = np.bincount(2 * np.asarray(y, dtype=int) + pred, minlength=4)
    return {"threshold": float(threshold), **{key: float(value) for key, value in
            _confusion_metrics(*counts).items()}}


def _interval(point, draws) -> dict:
    defined = draws[np.isfinite(draws)]
    lower, upper = np.quantile(defined, [0.025, 0.975]) if len(defined) else (None, None)
    return {
        "estimate": float(point) if np.isfinite(point) else None,
        "lower": float(lower) if lower is not None else None,
        "upper": float(upper) if upper is not None else None,
        "defined_resamples": len(defined),
    }


def compare_at_thresholds(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_new: np.ndarray,
    *,
    threshold_ref: float,
    threshold_new: float,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Evaluate paired validation scores using fixed, development-derived cutoffs.

    Outcome and both probabilities must be aligned in the same patient order.
    The joint count bootstrap is equivalent to resampling those patients together.
    Undefined ratios (including unbounded diagnostic ORs) are reported as null.
    """
    y = np.asarray(y)
    scores = [np.asarray(p, dtype=float) for p in (p_ref, p_new)]
    thresholds = (float(threshold_ref), float(threshold_new))
    if y.ndim != 1 or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("outcomes must be a vector containing both binary classes")
    for p in scores:
        if p.shape != y.shape or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
            raise ValueError("each probability vector must align with outcomes and lie in [0, 1]")
    if not all(0 <= cutoff <= 1 for cutoff in thresholds):
        raise ValueError("thresholds must lie in [0, 1]")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    ref, new = (p >= cutoff for p, cutoff in zip(scores, thresholds))
    counts = np.bincount(4 * y.astype(int) + 2 * ref + new, minlength=8).reshape(2, 2, 2)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(y), counts.ravel() / len(y), size=n_boot).reshape(-1, 2, 2, 2)
    point_ref = _confusion_metrics(*counts.sum(axis=2).ravel())
    point_new = _confusion_metrics(*counts.sum(axis=1).ravel())
    draw_ref = _confusion_metrics(*draws.sum(axis=3).reshape(-1, 4).T)
    draw_new = _confusion_metrics(*draws.sum(axis=2).reshape(-1, 4).T)
    metrics = {}
    for name, cutoff, point, bootstrap in (
        ("reference", thresholds[0], point_ref, draw_ref),
        ("new", thresholds[1], point_new, draw_new),
    ):
        metrics[name] = {
            "threshold": cutoff,
            **{key: _interval(value, bootstrap[key]) for key, value in point.items()},
        }

    # Sensitivity uses positive calls among events. Specificity uses negative
    # calls among non-events, so the discordant cells have the opposite order.
    tests = {}
    for name, outcome, correct in (("sensitivity", 1, 1), ("specificity", 0, 0)):
        n01 = int(counts[outcome, 1 - correct, correct])
        n10 = int(counts[outcome, correct, 1 - correct])
        tests[name] = {
            "n": int(counts[outcome].sum()),
            "reference_incorrect_new_correct": n01,
            "reference_correct_new_incorrect": n10,
            "p_value": float(stats.binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0,
        }

    # Two-category NRI is the sum of the differences in sensitivity and specificity.
    nri = point_new["sensitivity"] - point_ref["sensitivity"]
    nri += point_new["specificity"] - point_ref["specificity"]
    bootstrap_nri = draw_new["sensitivity"] - draw_ref["sensitivity"]
    bootstrap_nri += draw_new["specificity"] - draw_ref["specificity"]
    return {
        "models": metrics,
        "mcnemar": tests,
        "categorical_nri": _interval(nri, bootstrap_nri),
        "bootstrap_resamples": n_boot,
    }


def operating_point(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    best = int(np.argmax(tpr - fpr))
    # ROC's infinite sentinel predicts no events. In a Youden tie, prefer an
    # attainable probability cutoff before applying the existing ROC order.
    if not np.isfinite(thresholds[best]):
        best = 1 + int(np.argmax((tpr - fpr)[1:]))
    return _metrics_at_threshold(y, p, float(thresholds[best]))


def evaluate_model(
    y: np.ndarray, p: np.ndarray, *, threshold: float | None = None, seed: int = 0
) -> dict:
    disc = auroc_ci(y, p, seed=seed)
    cal = calibration_metrics(y, p)
    point = {} if threshold is None else _metrics_at_threshold(y, p, threshold)
    return {**disc.as_dict(), **cal.as_dict(), **point}


def subgroup_table(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_new: np.ndarray,
    subgroups: dict[str, np.ndarray],
    *,
    min_size: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests

    rows = []
    for name, mask in subgroups.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError(f"subgroup {name!r} must align with validation outcomes")
        yy = y[mask]
        if len(yy) < min_size or min(np.count_nonzero(yy == 0), np.count_nonzero(yy == 1)) < 2:
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
