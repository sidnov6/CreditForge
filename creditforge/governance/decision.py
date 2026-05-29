"""Decisioning — cost-based PD threshold + risk bands (Part 10.5).

The approve/decline cutoff is a *business* decision made explicit: pick the PD
threshold that minimizes expected misclassification cost, where a false negative
(approving a defaulter) costs `cost_fn` and a false positive (declining a good
borrower) costs `cost_fp`. Risk bands map the scaled credit score to rating
grades from config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config


def default_threshold(pd_hat, y_true, cfg: Config | None = None) -> float:
    """PD cutoff minimizing expected cost (cost_fn·FN + cost_fp·FP)."""
    cfg = cfg or load_config()
    cost_fn = float(cfg.decision.cost_fn)
    cost_fp = float(cfg.decision.cost_fp)
    p = np.asarray(pd_hat, float)
    y = np.asarray(y_true, int)

    candidates = np.quantile(p, np.linspace(0.01, 0.99, 99))
    best_thr, best_cost = candidates[0], np.inf
    for thr in candidates:
        approve = p <= thr
        fn = int(((y == 1) & approve).sum())      # approved a defaulter
        fp = int(((y == 0) & ~approve).sum())      # declined a good borrower
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr


def assign_band(score, cfg: Config | None = None) -> "np.ndarray | str":
    """Map credit score -> rating grade using configured band floors."""
    cfg = cfg or load_config()
    # Ascending by floor so each higher-grade floor overwrites the lower grade:
    # every loan ends up in the highest band it qualifies for.
    bands = sorted(cfg.decision.bands, key=lambda b: float(b["min_score"]))
    scalar = np.isscalar(score)
    s = np.atleast_1d(np.asarray(score, float))
    out = np.empty(s.shape, dtype=object)
    out[:] = bands[0]["name"]
    for b in bands:
        out[s >= float(b["min_score"])] = b["name"]
    return out[0] if scalar else out


def decide(pd_hat, threshold: float) -> "np.ndarray | str":
    p = np.asarray(pd_hat, float)
    decision = np.where(p <= threshold, "approve", "decline")
    return decision.item() if decision.ndim == 0 else decision


def decision_summary(scored: pd.DataFrame, pd_col: str, threshold: float,
                     cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    p = scored[pd_col].to_numpy()
    approve = p <= threshold
    bands = assign_band(scored["credit_score"].to_numpy(), cfg)
    band_counts = pd.Series(bands).value_counts().to_dict()
    return {
        "threshold": float(threshold),
        "approval_rate": float(approve.mean()),
        "approved_default_rate": float(scored.loc[approve, "default_12m"].mean()),
        "declined_default_rate": float(scored.loc[~approve, "default_12m"].mean()),
        "band_distribution": {k: int(v) for k, v in band_counts.items()},
    }
