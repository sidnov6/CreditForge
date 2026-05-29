"""Training orchestrator — fit + calibrate both PD models, score the OOT test set.

Produces the central artifact `artifacts/test_scored.parquet` consumed by the
validation suite, governance, serving, and the dashboard. Calibration is fit on
the held-out calibration vintages; everything reported downstream is on the
strictly out-of-time test vintages.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from creditforge.config import Config, load_config, run_stamp, set_global_seed
from creditforge.dataset import feature_columns
from creditforge.models import calibration, challenger, ead as ead_mod, el as el_mod
from creditforge.models import lgd as lgd_mod, scorecard
from creditforge.pipeline.split import Split, make_split


def _gini(y, p) -> float:
    return float(2 * roc_auc_score(y, p) - 1)


def train_all(cfg: Config | None = None, split: Split | None = None) -> Path:
    cfg = cfg or load_config()
    set_global_seed()
    split = split or make_split(cfg=cfg)
    feats = feature_columns()

    # ---- PD model A: regulatory scorecard --------------------------------
    sc = scorecard.fit_scorecard(split.train, cfg)
    scorecard.save(sc, cfg)
    sc_raw = {k: sc.predict_proba(getattr(split, k)[feats])[:, 1]
              for k in ("calib", "test")}

    # ---- PD model B: ML challenger ---------------------------------------
    ch = challenger.fit_challenger(split.train, split.calib, cfg)
    challenger.save(ch, cfg)
    ch_raw = {k: challenger.predict_pd(ch, getattr(split, k)) for k in ("calib", "test")}

    # ---- Calibrators (fit on calib vintages only) ------------------------
    sc_cal = calibration.fit_calibrator(sc_raw["calib"], split.calib["default_12m"], cfg)
    ch_cal = calibration.fit_calibrator(ch_raw["calib"], split.calib["default_12m"], cfg)
    calibration.save(sc_cal, "scorecard", cfg)
    calibration.save(ch_cal, "challenger", cfg)

    # ---- LGD (two-stage) + EAD -> Expected Loss --------------------------
    lgd_model = lgd_mod.fit_lgd(split.train, cfg)
    lgd_mod.save(lgd_model, cfg)

    # ---- Score the out-of-time test set ----------------------------------
    test = split.test
    pd_cal = ch_cal.transform(ch_raw["test"])          # challenger is champion for EL
    lgd_hat = lgd_model.predict(test)
    ead_hat = ead_mod.estimate_ead(test, cfg)
    scored = pd.DataFrame({
        "loan_id": test["loan_id"].to_numpy(),
        "vintage": test["vintage"].to_numpy(),
        "default_12m": test["default_12m"].astype(int).to_numpy(),
        "borrower_race": test["borrower_race"].to_numpy(),
        "ead_realized": test["ead"].to_numpy(),
        "lgd_realized": test["lgd"].to_numpy(),
        "pd_scorecard_raw": sc_raw["test"],
        "pd_scorecard": sc_cal.transform(sc_raw["test"]),
        "pd_challenger_raw": ch_raw["test"],
        "pd_challenger": pd_cal,
        "credit_score": sc.score(test[feats]),
        "lgd_hat": lgd_hat,
        "ead": ead_hat,
        "expected_loss": el_mod.expected_loss(pd_cal, lgd_hat, ead_hat),
    })
    out = cfg.path("artifacts") / "test_scored.parquet"
    scored.to_parquet(out, index=False)

    # Snapshot the training-time score+feature baseline for drift monitoring, so
    # the baseline always matches the model just trained (never goes stale).
    from creditforge.monitoring.drift import snapshot_baseline
    snapshot_baseline(cfg)

    metrics = {
        "stamp": run_stamp(),
        "split": split.summary(),
        "gini": {
            "scorecard": _gini(scored["default_12m"], scored["pd_scorecard"]),
            "challenger": _gini(scored["default_12m"], scored["pd_challenger"]),
        },
        "best_iteration_challenger": int(ch.best_iteration_ or ch.n_estimators),
        "expected_loss": {
            "portfolio_el": float(scored["expected_loss"].sum()),
            "total_ead": float(scored["ead"].sum()),
            "el_rate_bps": float(scored["expected_loss"].sum() / scored["ead"].sum() * 1e4),
            "mean_pd": float(scored["pd_challenger"].mean()),
            "mean_lgd_hat": float(scored["lgd_hat"].mean()),
        },
    }
    (cfg.path("artifacts") / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    # MLflow tracking (best-effort) — params, metrics, artifacts for reproducibility
    from creditforge.tracking import log_run
    with log_run("train", cfg) as logger:
        if logger:
            logger.params(calibration=cfg.calibration.method,
                          challenger_lr=cfg.challenger.learning_rate,
                          oot_test_start=cfg.target.oot.test_vintage_start)
            logger.metrics(gini_scorecard=metrics["gini"]["scorecard"],
                           gini_challenger=metrics["gini"]["challenger"],
                           portfolio_el=metrics["expected_loss"]["portfolio_el"],
                           el_rate_bps=metrics["expected_loss"]["el_rate_bps"])
            logger.artifacts(cfg.path("artifacts"))

    print(f"[train] OOT Gini  scorecard={metrics['gini']['scorecard']:.4f}  "
          f"challenger={metrics['gini']['challenger']:.4f}")
    print(f"[train] scored {len(scored):,} OOT loans -> {out}")
    return out


if __name__ == "__main__":
    train_all()
