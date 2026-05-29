"""Calibration validation — are the probabilities right? (Part 7)

A model can have great Gini and terrible calibration; we check both. The
Hosmer-Lemeshow test, a reliability curve, and per-band predicted-vs-observed
default rates. Run on the out-of-time set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def reliability_curve(y_true, p_hat, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per probability bin."""
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(p_hat, float)})
    # equal-frequency bins on predicted PD
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins,
                        labels=range(1, n_bins + 1))
    out = (df.groupby("bin", observed=True)
             .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
             .reset_index())
    out["abs_error"] = (out["predicted"] - out["observed"]).abs()
    return out


def hosmer_lemeshow(y_true, p_hat, n_groups: int = 10) -> dict:
    """HL goodness-of-fit test. Large p-value => calibration not rejected."""
    y = np.asarray(y_true, float)
    p = np.asarray(p_hat, float)
    order = np.argsort(p)
    y, p = y[order], p[order]
    groups = np.array_split(np.arange(len(y)), n_groups)
    hl = 0.0
    for idx in groups:
        if len(idx) == 0:
            continue
        obs1 = y[idx].sum()
        exp1 = p[idx].sum()
        n = len(idx)
        obs0 = n - obs1
        exp0 = n - exp1
        for obs, exp in ((obs1, exp1), (obs0, exp0)):
            if exp > 1e-9:
                hl += (obs - exp) ** 2 / exp
    dof = max(n_groups - 2, 1)
    p_value = float(chi2.sf(hl, dof))
    return {"hl_statistic": float(hl), "dof": dof, "p_value": p_value}


def expected_calibration_error(y_true, p_hat, n_bins: int = 10) -> float:
    """ECE: sample-weighted mean |predicted - observed| across bins."""
    rc = reliability_curve(y_true, p_hat, n_bins)
    w = rc["n"] / rc["n"].sum()
    return float((w * rc["abs_error"]).sum())


def calibration_report(y_true, p_hat, n_bins: int = 10) -> dict:
    rc = reliability_curve(y_true, p_hat, n_bins)
    return {
        "hosmer_lemeshow": hosmer_lemeshow(y_true, p_hat, n_bins),
        "ece": expected_calibration_error(y_true, p_hat, n_bins),
        "max_band_error": float(rc["abs_error"].max()),
        "reliability": rc.to_dict(orient="records"),
    }
