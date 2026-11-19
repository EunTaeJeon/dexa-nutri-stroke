import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from dexa_nutri_stroke.interpret import _crossing, dependence, global_importance, shap_values_for
from dexa_nutri_stroke.model import calibrated_estimator


@pytest.fixture(scope="module")
def calibrated_model():
    i = np.arange(48)
    # The features are close enough for the default masker to miss tree splits.
    X = pd.DataFrame({
        "a": 1.0 + (i * 7 % 47) * 2e-7,
        "b": 1.0 + (i * 11 % 43) * 2e-7,
    })
    y = ((i * 7 % 47 > 19) ^ (i * 11 % 43 > 28) ^ (i % 5 == 0)).astype(int)
    pipeline = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=3, random_state=1))
    model = calibrated_estimator(pipeline, n_folds=3, seed=7).fit(X, y)
    return model, X


def test_shap_reconstructs_the_calibrated_ensemble_probability(calibrated_model):
    model, background = calibrated_model
    X = background.iloc[::5].copy()
    X.iloc[0, 0] = np.nan
    reference = model.predict_proba(background)[:, 1].mean()
    prediction = model.predict_proba(X)[:, 1]
    first_fold = model.calibrated_classifiers_[0].predict_proba(X)[:, 1]
    assert len(model.calibrated_classifiers_) == 3
    assert not np.allclose(prediction, first_fold)
    assert np.ptp(prediction) > 0

    values, predictors = shap_values_for(model, X, background=background[["b", "a"]])
    pd.testing.assert_frame_equal(predictors, X)
    np.testing.assert_allclose(reference + values.sum(axis=1), prediction, rtol=0, atol=1e-12)
    # With two features, their four coalitions give exact Shapley values.
    for position, (_, row) in enumerate(X.iterrows()):
        only_a = model.predict_proba(background.assign(a=row["a"]))[:, 1].mean()
        only_b = model.predict_proba(background.assign(b=row["b"]))[:, 1].mean()
        exact = 0.5 * np.array([
            only_a - reference + prediction[position] - only_b,
            only_b - reference + prediction[position] - only_a,
        ])
        np.testing.assert_allclose(values[position], exact, rtol=0, atol=1e-12)

    importance = global_importance(model, X, background=background).set_index("feature")
    np.testing.assert_allclose(importance.loc[X.columns, "mean_abs_shap"], np.abs(values).mean(axis=0))
    curve = dependence(model, X, "a", background=background)
    np.testing.assert_array_equal(curve.x, X["a"].dropna())
    np.testing.assert_allclose(curve.shap, values[X["a"].notna(), 0])


def test_uncalibrated_tree_shap_uses_the_supplied_background(calibrated_model):
    model, background = calibrated_model
    tree = model.calibrated_classifiers_[0].estimator
    X = background.iloc[::5]
    values, predictors = shap_values_for(tree, X, background=background)
    reference = tree.predict_proba(background)[:, 1].mean()
    np.testing.assert_array_equal(predictors, X.to_numpy())
    np.testing.assert_allclose(reference + values.sum(axis=1), tree.predict_proba(X)[:, 1], atol=1e-8)


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        ([1.0, -1.0, -1.0, -1.0], 0.5),
        ([1.0, 0.0, -1.0, -2.0], 1.0),
        ([1.0, 0.0, 1.0, 2.0], np.nan),
        ([1.0, -1.0, 1.0, -1.0], np.nan),
        ([1.0, 0.0, 0.0, -1.0], np.nan),
    ],
)
def test_zero_crossing_requires_a_unique_change_of_sign(curve, expected):
    actual = _crossing(np.arange(len(curve)), np.array(curve))
    np.testing.assert_allclose(actual, expected)
