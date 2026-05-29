"""PD model A — the regulatory WoE/IV + logistic scorecard.

The technique banks actually use and regulators accept: optimal monotonic
binning, Weight-of-Evidence encoding, Information-Value feature selection,
logistic regression on WoE, scaled to points (PDO/odds). Interpretable and
monotonic by construction; the WoE binning also tames outliers and missing
values (a robustness win).

Produces true PDs via `predict_proba` and a points score via `score` (higher
score = lower risk). Calibration is handled separately (Part 3.2).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from creditforge.config import Config, load_config, run_stamp
from creditforge.dataset import categorical_columns, feature_columns


def fit_scorecard(train: pd.DataFrame, cfg: Config | None = None) -> Scorecard:
    cfg = cfg or load_config()
    sc_cfg = cfg.scorecard
    features = feature_columns()
    categorical = categorical_columns()

    X = train[features]
    y = train["default_12m"].astype(int)

    # IV-based selection: DROP near-useless features (IV < iv_min). Features
    # above iv_max are *flagged for leakage review*, not silently dropped — a
    # legitimately strong bureau score (FICO) earns a high IV honestly. The
    # review flag is surfaced via `high_iv_review()`; an analyst keeps or cuts.
    selection = {"iv": {"min": float(sc_cfg.iv_min)}}
    binning = BinningProcess(
        variable_names=features,
        categorical_variables=categorical,
        min_bin_size=float(sc_cfg.min_bin_size),
        selection_criteria=selection,
    )

    estimator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")

    # Points-to-double-odds scaling — the industry-standard scorecard scaling.
    scaling = {"pdo": float(sc_cfg.pdo), "odds": float(sc_cfg.base_odds),
               "scorecard_points": float(sc_cfg.base_score)}
    scorecard = Scorecard(
        binning_process=binning,
        estimator=estimator,
        scaling_method="pdo_odds",
        scaling_method_params=scaling,
        reverse_scorecard=False,  # convention: higher score = lower PD (lower risk)
    )
    scorecard.fit(X, y)
    return scorecard


def information_value(scorecard: Scorecard) -> pd.DataFrame:
    """IV table: which features were predictive (and which were dropped)."""
    summ = scorecard.binning_process_.summary()
    cols = [c for c in ("name", "iv", "selected") if c in summ.columns]
    return summ[cols].sort_values("iv", ascending=False).reset_index(drop=True)


def high_iv_review(scorecard: Scorecard, cfg: Config | None = None) -> list[str]:
    """Features with IV above the suspicion threshold — flagged for leakage review."""
    cfg = cfg or load_config()
    iv = information_value(scorecard)
    flagged = iv[iv["iv"] > float(cfg.scorecard.iv_max)]["name"].tolist()
    return flagged


def save(scorecard: Scorecard, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    out = cfg.path("artifacts") / "scorecard.joblib"
    joblib.dump({"model": scorecard, "stamp": run_stamp()}, out)
    return out


def load(cfg: Config | None = None) -> Scorecard:
    cfg = cfg or load_config()
    return joblib.load(cfg.path("artifacts") / "scorecard.joblib")["model"]


if __name__ == "__main__":
    from creditforge.pipeline.split import make_split

    split = make_split()
    sc = fit_scorecard(split.train)
    iv = information_value(sc)
    print("[scorecard] Information Value by feature:")
    print(iv.to_string(index=False))
    flagged = high_iv_review(sc)
    if flagged:
        print(f"[scorecard] HIGH-IV REVIEW (potential leakage, kept): {flagged}")
    save(sc)
    pd_hat = sc.predict_proba(split.test[feature_columns()])[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"[scorecard] OOT Gini = {2 * roc_auc_score(split.test['default_12m'], pd_hat) - 1:.4f}")
