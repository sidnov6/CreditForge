"""Drift monitoring (Part 10.4).

Scheduled PSI/CSI of an incoming batch's score + feature distributions vs the
training baseline, with threshold-based alerts. The baseline is snapshotted at
training time so monitoring is reproducible. Populates the dashboard Monitoring
screen and can run on a GitHub Actions cron.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from creditforge.config import Config, load_config, run_stamp
from creditforge.dataset import categorical_columns, numeric_columns
from creditforge.validation import stability as stab


def snapshot_baseline(cfg: Config | None = None) -> Path:
    """Persist the training-time score + feature baseline for drift comparison."""
    cfg = cfg or load_config()
    from creditforge.models import scorecard
    from creditforge.pipeline.split import make_split
    from creditforge.dataset import feature_columns

    train = make_split(cfg=cfg).train
    sc = scorecard.load(cfg)
    baseline = train[numeric_columns() + categorical_columns()].copy()
    baseline["credit_score"] = sc.score(train[feature_columns()])
    out = cfg.path("artifacts") / "baseline.parquet"
    baseline.to_parquet(out, index=False)
    return out


def monitor_batch(batch: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Compute score-PSI + feature-CSI of `batch` vs the training baseline."""
    cfg = cfg or load_config()
    bins = int(cfg.validation.psi_bins)
    baseline = pd.read_parquet(cfg.path("artifacts") / "baseline.parquet")

    score_psi = (stab.psi(baseline["credit_score"], batch["credit_score"], bins)
                 if "credit_score" in batch else None)
    feats = [f for f in (numeric_columns() + categorical_columns()) if f in batch]
    csi = stab.csi_by_feature(baseline, batch, feats, bins)
    alerts = []
    if score_psi is not None and stab.psi_status(score_psi) != "stable":
        alerts.append({"signal": "score", "psi": round(score_psi, 4),
                       "status": stab.psi_status(score_psi)})
    for rec in csi[csi["status"] != "stable"].to_dict(orient="records"):
        alerts.append({"signal": rec["feature"], "psi": round(rec["csi"], 4),
                       "status": rec["status"]})

    return {
        "stamp": run_stamp(),
        "n_batch": int(len(batch)),
        "score_psi": (round(float(score_psi), 4) if score_psi is not None else None),
        "score_status": (stab.psi_status(score_psi) if score_psi is not None else None),
        "feature_csi": csi.to_dict(orient="records"),
        "alerts": alerts,
        "healthy": len(alerts) == 0,
    }


def run_monitoring(cfg: Config | None = None) -> dict:
    """Demo run: monitor the OOT scored set against the training baseline."""
    cfg = cfg or load_config()
    if not (cfg.path("artifacts") / "baseline.parquet").exists():
        snapshot_baseline(cfg)
    scored = pd.read_parquet(cfg.path("artifacts") / "test_scored.parquet")
    # the scored set has credit_score; pull features from the OOT split too
    from creditforge.pipeline.split import make_split
    test = make_split(cfg=cfg).test.merge(
        scored[["loan_id", "credit_score"]], on="loan_id")
    report = monitor_batch(test, cfg)
    (cfg.path("artifacts") / "monitoring_report.json").write_text(
        json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    r = run_monitoring()
    print(f"[monitor] score PSI={r['score_psi']} ({r['score_status']}) | "
          f"{len(r['alerts'])} alert(s) | healthy={r['healthy']}")
    for a in r["alerts"]:
        print(f"    ALERT {a['signal']}: PSI={a['psi']} ({a['status']})")
