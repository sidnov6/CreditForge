"""Discrimination — can the model separate good from bad? (Part 7)

Gini (= 2·AUC − 1, the bank-standard headline), KS, and a gains/lift table by
score band. Reported on the out-of-time set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y_true, y_score) -> float:
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def ks_statistic(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def roc_points(y_true, y_score) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def gains_table(y_true, y_score, n_bands: int = 10) -> pd.DataFrame:
    """Decile gains/lift table, riskiest band first."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    # rank by descending PD: band 1 = highest risk
    df["band"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                         q=n_bands, labels=range(1, n_bands + 1))
    base_rate = df["y"].mean()
    g = (df.groupby("band", observed=True)
           .agg(n=("y", "size"), defaults=("y", "sum"),
                avg_pd=("p", "mean"))
           .reset_index())
    g["default_rate"] = g["defaults"] / g["n"]
    g["lift"] = g["default_rate"] / base_rate
    g["cum_defaults"] = g["defaults"].cumsum()
    g["cum_capture_rate"] = g["cum_defaults"] / df["y"].sum()
    return g


def discrimination_report(y_true, y_score, n_bands: int = 10) -> dict:
    return {
        "auc": auc(y_true, y_score),
        "gini": gini(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "gains": gains_table(y_true, y_score, n_bands).to_dict(orient="records"),
    }
