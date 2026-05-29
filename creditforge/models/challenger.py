"""PD model B — the ML challenger (LightGBM).

Quantifies the accuracy ceiling and what the interpretable scorecard leaves on
the table. Native categorical handling, class-imbalance via `scale_pos_weight`
(not naive oversampling — Part 10.5), early stopping on the held-out calibration
vintages. Raw probabilities here are *uncalibrated*; calibration is Part 3.2.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from creditforge.config import Config, load_config, run_stamp
from creditforge.dataset import categorical_columns, feature_columns


def _as_model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure categorical dtype so LightGBM uses native categorical splits."""
    X = X[feature_columns()].copy()
    for c in categorical_columns():
        X[c] = X[c].astype("category")
    return X


def fit_challenger(train: pd.DataFrame, valid: pd.DataFrame,
                   cfg: Config | None = None) -> lgb.LGBMClassifier:
    cfg = cfg or load_config()
    c = cfg.challenger

    y = train["default_12m"].astype(int)
    X = _as_model_frame(train)
    Xv, yv = _as_model_frame(valid), valid["default_12m"].astype(int)

    # Imbalance handling: reweighting the positive class distorts ranking and is
    # off by default (probability levels are fixed by calibration downstream).
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = (neg / max(pos, 1)) if bool(c.use_scale_pos_weight) else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=int(c.n_estimators),
        learning_rate=float(c.learning_rate),
        num_leaves=int(c.num_leaves),
        min_child_samples=int(c.min_child_samples),
        reg_lambda=float(c.reg_lambda),
        subsample=float(c.subsample),
        colsample_bytree=float(c.colsample_bytree),
        scale_pos_weight=scale_pos_weight,
        random_state=int(cfg.run.seed),
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X, y,
        eval_set=[(Xv, yv)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(int(c.early_stopping_rounds), verbose=False)],
    )
    return model


def predict_pd(model: lgb.LGBMClassifier, X: pd.DataFrame) -> "pd.Series":
    return model.predict_proba(_as_model_frame(X))[:, 1]


def save(model: lgb.LGBMClassifier, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    out = cfg.path("artifacts") / "challenger.joblib"
    joblib.dump({"model": model, "stamp": run_stamp(),
                 "features": feature_columns(),
                 "categorical": categorical_columns()}, out)
    return out


def load(cfg: Config | None = None) -> lgb.LGBMClassifier:
    cfg = cfg or load_config()
    return joblib.load(cfg.path("artifacts") / "challenger.joblib")["model"]


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    from creditforge.pipeline.split import make_split

    split = make_split()
    model = fit_challenger(split.train, split.calib)
    save(model)
    pd_hat = predict_pd(model, split.test)
    gini = 2 * roc_auc_score(split.test["default_12m"], pd_hat) - 1
    print(f"[challenger] best_iteration={model.best_iteration_} | OOT Gini = {gini:.4f}")
