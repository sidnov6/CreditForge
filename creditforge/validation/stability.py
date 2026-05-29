"""Stability — does it hold across time / populations? (Part 7 & 10.4)

PSI on the score distribution between a baseline and a target population, and
CSI per feature. Canonical thresholds: < 0.10 stable, 0.10-0.25 watch,
> 0.25 unstable. The same PSI engine powers drift monitoring (Part 10.4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)  # guard against ties collapsing bins
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float, watch: float = 0.10, unstable: float = 0.25) -> str:
    if value < watch:
        return "stable"
    if value < unstable:
        return "watch"
    return "unstable"


def psi_by_vintage(df: pd.DataFrame, score_col: str, vintage_col: str = "vintage",
                   bins: int = 10) -> pd.DataFrame:
    """PSI of the score distribution in each vintage vs the earliest vintage."""
    vintages = sorted(df[vintage_col].unique())
    baseline = df[df[vintage_col] == vintages[0]][score_col].to_numpy()
    rows = []
    for v in vintages:
        actual = df[df[vintage_col] == v][score_col].to_numpy()
        val = psi(baseline, actual, bins) if v != vintages[0] else 0.0
        rows.append({"vintage": v, "psi": val, "status": psi_status(val), "n": len(actual)})
    return pd.DataFrame(rows)


def csi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Characteristic Stability Index for a single feature (numeric or categorical)."""
    if expected.dtype.name in ("category", "object") or expected.nunique() <= bins:
        e = expected.value_counts(normalize=True)
        a = actual.value_counts(normalize=True)
        cats = e.index.union(a.index)
        e = e.reindex(cats).fillna(0).clip(lower=1e-6)
        a = a.reindex(cats).fillna(0).clip(lower=1e-6)
        return float(np.sum((a - e) * np.log(a / e)))
    return psi(expected.to_numpy(), actual.to_numpy(), bins)


def csi_by_feature(expected: pd.DataFrame, actual: pd.DataFrame,
                   features: list[str], bins: int = 10) -> pd.DataFrame:
    rows = []
    for f in features:
        val = csi(expected[f], actual[f], bins)
        rows.append({"feature": f, "csi": val, "status": psi_status(val)})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
