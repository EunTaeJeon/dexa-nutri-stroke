from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dexa_nutri_stroke.data import Schema, split_once
from dexa_nutri_stroke.evaluate import auroc_ci, calibration_metrics, delong_test
from dexa_nutri_stroke.model import build_estimator, cross_val_probabilities
from dexa_nutri_stroke.preprocess import (
    DropHighMissing,
    IterativeImputerDF,
    WinsoriseOutliers,
)
from dexa_nutri_stroke.schema import default_schema
from make_synthetic_data import generate


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return generate(n=300, seed=1)


@pytest.fixture(scope="module")
def schema() -> Schema:
    return default_schema()


def test_split_is_reproducible(cohort, schema):
    a = split_once(cohort, schema, seed=7)
    b = split_once(cohort, schema, seed=7)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.valid, b.valid)


def test_split_partitions_the_cohort(cohort, schema):
    ds = split_once(cohort, schema, valid_size=0.4, seed=7)
    assert len(ds.train) + len(ds.valid) == len(cohort)


def test_split_preserves_event_rate(cohort, schema):
    ds = split_once(cohort, schema, valid_size=0.4, seed=7)
    overall = cohort[schema.outcome].mean()
    assert abs(ds.train[schema.outcome].mean() - overall) < 0.08
    assert abs(ds.valid[schema.outcome].mean() - overall) < 0.08


def test_drop_high_missing_applies_fitted_columns():
    train = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    other = pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 6.0]})

    step = DropHighMissing(threshold=0.2).fit(train)
    assert list(step.transform(other).columns) == ["b"]


def test_winsorise_clips_to_fitted_range():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    other = pd.DataFrame({"x": [-100.0, 100.0]})

    out = WinsoriseOutliers(contamination=0.0).fit(train).transform(other)
    assert out["x"].iloc[0] == pytest.approx(1.0)
    assert out["x"].iloc[1] == pytest.approx(5.0)


def test_imputer_ignores_y(cohort, schema):
    X = cohort[list(schema.full_features)]
    y = cohort[schema.outcome].to_numpy()

    a = IterativeImputerDF(categorical=schema.categorical, random_state=0).fit(X, y)
    b = IterativeImputerDF(categorical=schema.categorical, random_state=0).fit(X, None)
    pd.testing.assert_frame_equal(a.transform(X), b.transform(X))


def test_imputer_transform_is_row_wise(cohort, schema):
    X = cohort[list(schema.full_features)]
    fitted = IterativeImputerDF(categorical=schema.categorical, random_state=0).fit(X.iloc[:200])

    single = fitted.transform(X.iloc[200:201])
    batch = fitted.transform(X.iloc[200:])
    np.testing.assert_allclose(
        single.to_numpy(dtype=float), batch.iloc[:1].to_numpy(dtype=float), rtol=1e-9
    )


def test_estimator_contains_preprocessing_steps(cohort, schema):
    est = build_estimator(schema.categorical, {"n_estimators": 20}, random_state=0)
    assert "impute" in dict(est.named_steps)
    assert list(est.named_steps)[-1] == "classifier"


def test_cross_val_probabilities_are_valid(cohort, schema):
    est = build_estimator(schema.categorical, {"n_estimators": 20}, random_state=0)
    X = cohort[list(schema.full_features)]
    y = cohort[schema.outcome].to_numpy()

    proba = cross_val_probabilities(est, X, y, n_folds=3, seed=0)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_auroc_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 400)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.2, 400), 0.01, 0.99)

    d = auroc_ci(y, p, n_boot=300, seed=0)
    assert d.lower <= d.auroc <= d.upper
    assert d.auroc > 0.7


def test_delong_difference_is_zero_for_equal_predictions():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.4, 300)
    p = rng.random(300)
    assert delong_test(y, p, p)["difference"] == pytest.approx(0.0, abs=1e-12)


def test_delong_separates_unequal_models():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 500)
    good = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.15, 500), 0.01, 0.99)
    poor = rng.random(500)

    out = delong_test(y, good, poor)
    assert out["difference"] > 0
    assert out["p_value"] < 0.001


def test_calibration_metrics_on_sharp_predictions():
    y = np.array([0, 0, 1, 1] * 50)
    p = np.where(y == 1, 0.99, 0.01)

    cal = calibration_metrics(y, p)
    assert cal.brier < 0.01
    assert cal.ici < 0.05
