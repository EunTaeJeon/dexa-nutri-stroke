from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["Schema", "Dataset", "load_dataset", "split_once"]


@dataclass(frozen=True)
class Schema:
    outcome: str
    clinical: tuple[str, ...]
    body_composition: tuple[str, ...] = ()
    nutrition: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    stratify_by: tuple[str, ...] = ()

    @property
    def clin_features(self) -> list[str]:
        return list(self.clinical)

    @property
    def full_features(self) -> list[str]:
        return list(self.clinical) + list(self.body_composition) + list(self.nutrition)

    def features_for(self, model: str) -> list[str]:
        if model == "clin":
            return self.clin_features
        if model == "clin_dexa_nutri":
            return self.full_features
        raise ValueError(f"unknown model {model!r}; expected 'clin' or 'clin_dexa_nutri'")

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in [self.outcome, *self.full_features] if c not in df.columns]
        if missing:
            raise KeyError(f"columns absent from the data: {missing}")

        y = df[self.outcome].dropna().unique()
        if not set(y).issubset({0, 1}):
            raise ValueError(f"{self.outcome!r} must be binary 0/1, found {sorted(y)[:5]}")

        unknown_cat = set(self.categorical) - set(self.full_features)
        if unknown_cat:
            raise ValueError(f"categorical columns not among the features: {sorted(unknown_cat)}")

        absent_strat = [c for c in self.stratify_by if c not in df.columns]
        if absent_strat:
            raise KeyError(f"stratification columns absent from the data: {absent_strat}")


@dataclass
class Dataset:
    schema: Schema
    train: pd.DataFrame
    valid: pd.DataFrame
    _meta: dict = field(default_factory=dict)

    def xy(self, part: str, model: str) -> tuple[pd.DataFrame, np.ndarray]:
        df = self.train if part == "train" else self.valid
        features = self.schema.features_for(model)
        return df[features].copy(), df[self.schema.outcome].to_numpy(dtype=int)

    def __repr__(self) -> str:
        n_tr, n_va = len(self.train), len(self.valid)
        e_tr = int(self.train[self.schema.outcome].sum())
        e_va = int(self.valid[self.schema.outcome].sum())
        return (
            f"Dataset(train={n_tr} rows / {e_tr} events, "
            f"valid={n_va} rows / {e_va} events)"
        )


def load_dataset(path: str | Path, schema: Schema) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported extension {path.suffix!r}; use .csv or .parquet")

    schema.validate(df)

    before = len(df)
    df = df[df[schema.outcome].notna()].reset_index(drop=True)
    df[schema.outcome] = df[schema.outcome].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"[data] dropped {dropped} row(s) without an observed outcome")
    return df


def _stratification_key(df: pd.DataFrame, schema: Schema) -> pd.Series:
    parts = [df[schema.outcome].astype(str)]
    for col in schema.stratify_by:
        parts.append(df[col].astype("string").fillna("NA"))
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep="|")

    counts = key.value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        key = key.mask(key.isin(rare), df[schema.outcome].astype(str))
    return key


def split_once(
    df: pd.DataFrame,
    schema: Schema,
    *,
    valid_size: float = 0.4,
    seed: int = 20240930,
) -> Dataset:
    if not 0.0 < valid_size < 1.0:
        raise ValueError(f"valid_size must lie in (0, 1), got {valid_size}")

    key = _stratification_key(df, schema)
    train, valid = train_test_split(
        df,
        test_size=valid_size,
        stratify=key,
        shuffle=True,
        random_state=seed,
    )
    ds = Dataset(
        schema=schema,
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        _meta={"seed": seed, "valid_size": valid_size},
    )
    print(f"[data] {ds}")
    return ds
