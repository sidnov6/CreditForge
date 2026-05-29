"""SHAP explainability — global + local attributions (Part 8).

Global importance (what drives the model overall) and per-applicant local
attributions (why *this* decision), computed on the challenger via the fast
TreeExplainer. Local attributions feed the adverse-action reason codes.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from creditforge.config import Config, load_config
from creditforge.dataset import categorical_columns, feature_columns
from creditforge.models import challenger as ch_mod


def _model_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X[feature_columns()].copy()
    for c in categorical_columns():
        X[c] = X[c].astype("category")
    return X


def build_explainer(model=None, cfg: Config | None = None) -> shap.TreeExplainer:
    cfg = cfg or load_config()
    model = model or ch_mod.load(cfg)
    return shap.TreeExplainer(model)


def global_importance(explainer, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature — the global driver ranking."""
    sv = _shap_values(explainer, X)
    imp = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": feature_columns(), "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def local_attribution(explainer, x_row: pd.DataFrame) -> pd.DataFrame:
    """Signed SHAP contributions for one applicant (positive => raises PD)."""
    sv = _shap_values(explainer, x_row)[0]
    return (pd.DataFrame({"feature": feature_columns(),
                          "value": x_row[feature_columns()].iloc[0].astype(str).to_numpy(),
                          "shap": sv})
            .sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True))


def _shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    sv = explainer.shap_values(_model_frame(X))
    # LightGBM binary: TreeExplainer may return a list [class0, class1]
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv)


def save_global(cfg: Config | None = None) -> Path:
    """Precompute + persist global importance for the dashboard Governance screen."""
    cfg = cfg or load_config()
    from creditforge.pipeline.split import make_split
    expl = build_explainer(cfg=cfg)
    test = make_split(cfg=cfg).test
    sample = test.sample(min(2000, len(test)), random_state=int(cfg.run.seed))
    gi = global_importance(expl, sample)
    out = cfg.path("artifacts") / "shap_global.json"
    out.write_text(gi.to_json(orient="records"))
    return out


if __name__ == "__main__":
    p = save_global()
    print("[shap] global importance ->", p)
    import json
    print(json.dumps(json.loads(p.read_text())[:5], indent=2))
