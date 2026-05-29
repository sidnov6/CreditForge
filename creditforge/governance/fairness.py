"""Fairness / bias testing (Part 8).

Group metrics on the protected attribute (race), computed at the decision
threshold: selection-rate disparity (the 4/5ths / disparate-impact rule),
equal-opportunity difference (TPR gap), and per-group default-rate vs
approval-rate. The protected attribute is NEVER a model input — this measures
whether neutral features still produce disparate outcomes (proxy discrimination),
which the EU AI Act's high-risk regime expects us to surface and discuss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config


def _approve(pd_hat: np.ndarray, threshold: float) -> np.ndarray:
    """Approve if predicted PD is at/below the decision threshold."""
    return (pd_hat <= threshold).astype(int)


def group_metrics(df: pd.DataFrame, pd_col: str, threshold: float,
                  cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    attr = str(cfg.fairness.protected_attribute)
    d = df.copy()
    d["approved"] = _approve(d[pd_col].to_numpy(), threshold)
    d["y"] = d["default_12m"].astype(int)

    rows = []
    for grp, g in d.groupby(attr):
        actual_good = g[g["y"] == 0]
        tpr_good = (actual_good["approved"].mean() if len(actual_good) else np.nan)
        rows.append({
            "group": grp,
            "n": int(len(g)),
            "approval_rate": float(g["approved"].mean()),
            "default_rate": float(g["y"].mean()),
            "mean_pd": float(g[pd_col].mean()),
            # equal-opportunity basis: approval rate among truly good borrowers
            "tpr_good": float(tpr_good),
        })
    return pd.DataFrame(rows).sort_values("approval_rate", ascending=False).reset_index(drop=True)


def fairness_report(df: pd.DataFrame, pd_col: str, threshold: float,
                    cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    priv = str(cfg.fairness.privileged_group)
    gm = group_metrics(df, pd_col, threshold, cfg)

    by_group = gm.set_index("group")
    ref_rate = float(by_group.loc[priv, "approval_rate"]) if priv in by_group.index \
        else float(gm["approval_rate"].max())
    ref_tpr = float(by_group.loc[priv, "tpr_good"]) if priv in by_group.index \
        else float(gm["tpr_good"].max())

    gm = gm.assign(
        disparate_impact=lambda x: x["approval_rate"] / ref_rate,
        equal_opportunity_diff=lambda x: x["tpr_good"] - ref_tpr,
    )
    di_min = float(gm["disparate_impact"].min())
    return {
        "threshold": float(threshold),
        "privileged_group": priv,
        "di_ratio_min_observed": di_min,
        "di_ratio_floor": float(cfg.fairness.di_ratio_min),
        "passes_four_fifths": bool(di_min >= float(cfg.fairness.di_ratio_min)),
        "max_equal_opportunity_diff": float(gm["equal_opportunity_diff"].abs().max()),
        "groups": gm.to_dict(orient="records"),
    }


if __name__ == "__main__":
    from creditforge.governance.decision import default_threshold

    cfg = load_config()
    scored = pd.read_parquet(cfg.path("artifacts") / "test_scored.parquet")
    thr = default_threshold(scored["pd_challenger"], scored["default_12m"], cfg)
    rep = fairness_report(scored, "pd_challenger", thr, cfg)
    print(f"[fairness] threshold={thr:.4f} | min disparate-impact ratio="
          f"{rep['di_ratio_min_observed']:.3f} (4/5ths floor {rep['di_ratio_floor']}) "
          f"-> {'PASS' if rep['passes_four_fifths'] else 'REVIEW'}")
    for g in rep["groups"]:
        print(f"  {g['group']:9s} approve={g['approval_rate']:.3f} "
              f"DI={g['disparate_impact']:.3f} EOdiff={g['equal_opportunity_diff']:+.3f}")
