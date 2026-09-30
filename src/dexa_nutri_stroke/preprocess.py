from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer

__all__ = ["DropHighMissing", "WinsoriseOutliers", "IterativeImputerDF", "build_preprocessor"]


class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        rates = X.isna().mean()
        self.columns_ = list(rates[rates <= self.threshold].index)
        self.dropped_ = list(rates[rates > self.threshold].index)
        if not self.columns_:
            raise ValueError("every column exceeds the missingness threshold")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class WinsoriseOutliers(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical: tuple[str, ...] = (),
        contamination: float = 0.05,
        random_state: int = 0,
    ):
        self.categorical = categorical
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.continuous_ = [c for c in X.columns if c not in set(self.categorical)]
        self.bounds_: dict[str, tuple[float, float]] = {}
        if not self.continuous_:
            return self

        block = X[self.continuous_]
        filled = block.fillna(block.median(numeric_only=True)).fillna(0.0)

        inlier_mask = np.ones(len(filled), dtype=bool)
        if len(filled) >= 20 and 0.0 < self.contamination < 0.5:
            forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
            )
            inlier_mask = forest.fit_predict(filled) == 1
            if inlier_mask.sum() < max(10, int(0.5 * len(filled))):
                inlier_mask = np.ones(len(filled), dtype=bool)

        inliers = block.loc[inlier_mask]
        for col in self.continuous_:
            lo, hi = inliers[col].min(), inliers[col].max()
            if pd.isna(lo) or pd.isna(hi):
                lo, hi = block[col].min(), block[col].max()
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in out.columns and pd.notna(lo) and pd.notna(hi):
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features), dtype=object)


class IterativeImputerDF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical: tuple[str, ...] = (),
        max_iter: int = 10,
        n_nearest_features: int | None = 20,
        random_state: int = 0,
    ):
        self.categorical = categorical
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        n_near = self.n_nearest_features
        if n_near is not None:
            n_near = min(n_near, X.shape[1])
        self.imputer_ = IterativeImputer(
            max_iter=self.max_iter,
            n_nearest_features=n_near,
            random_state=self.random_state,
            sample_posterior=False,
            keep_empty_features=True,
        )
        self.imputer_.fit(X.to_numpy(dtype=float))
        self._cat_levels_ = {
            c: np.sort(X[c].dropna().unique()) for c in self.categorical if c in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.imputer_.transform(X[self.columns_].to_numpy(dtype=float))
        out = pd.DataFrame(arr, columns=self.columns_, index=X.index)
        for col, levels in self._cat_levels_.items():
            if len(levels) == 0:
                continue
            values = out[col].to_numpy()
            nearest = levels[np.abs(values[:, None] - levels[None, :]).argmin(axis=1)]
            out[col] = nearest
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def build_preprocessor(
    categorical: tuple[str, ...] = (),
    *,
    missing_threshold: float = 0.20,
    contamination: float = 0.05,
    random_state: int = 0,
) -> list[tuple[str, BaseEstimator]]:
    return [
        ("drop_sparse", DropHighMissing(threshold=missing_threshold)),
        (
            "winsorise",
            WinsoriseOutliers(
                categorical=categorical,
                contamination=contamination,
                random_state=random_state,
            ),
        ),
        (
            "impute",
            IterativeImputerDF(categorical=categorical, random_state=random_state),
        ),
    ]
