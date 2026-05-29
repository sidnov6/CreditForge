"""Shared dataset access — one place that knows the feature manifest.

Models and serving import from here so they always agree on which columns are
features (and which are off-limits, e.g. the protected attribute and the LGD
fields). Keeps the leakage guarantee in a single, auditable spot.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from creditforge.config import Config, load_config


@lru_cache(maxsize=1)
def feature_manifest() -> dict:
    cfg = load_config()
    path = cfg.path("gold") / "feature_manifest.json"
    return json.loads(path.read_text())


def feature_columns() -> list[str]:
    m = feature_manifest()
    return list(m["numeric"]) + list(m["categorical"])


def numeric_columns() -> list[str]:
    return list(feature_manifest()["numeric"])


def categorical_columns() -> list[str]:
    return list(feature_manifest()["categorical"])


def target_column() -> str:
    return feature_manifest()["target"]


def load_gold(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(cfg.path("gold") / "feature_matrix.parquet")


def Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the model feature matrix X and the target y."""
    return df[feature_columns()].copy(), df[target_column()].astype(int)
