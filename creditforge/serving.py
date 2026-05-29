"""Serving bundle — load every artifact once and score one applicant end-to-end.

Encapsulates the production scoring path: build the leakage-safe features from
raw inputs, run scorecard + challenger, calibrate, scale to a credit score and
risk band, predict LGD/EAD, compute Expected Loss, apply the cost-based decision,
and attach SHAP-driven adverse-action reason codes. Used by the FastAPI service.
"""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config
from creditforge.dataset import (categorical_columns, feature_columns,
                                 numeric_columns)
from creditforge.governance import decision, reason_codes, shap_explain
from creditforge.models import calibration, challenger, ead as ead_mod
from creditforge.models import el as el_mod, lgd as lgd_mod, scorecard


class ModelBundle:
    """Lazily loads all trained artifacts and serves single-applicant scoring."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()

    @cached_property
    def scorecard(self):
        return scorecard.load(self.cfg)

    @cached_property
    def challenger(self):
        return challenger.load(self.cfg)

    @cached_property
    def cal_sc(self):
        return calibration.load("scorecard", self.cfg)

    @cached_property
    def cal_ch(self):
        return calibration.load("challenger", self.cfg)

    @cached_property
    def lgd(self):
        return lgd_mod.load(self.cfg)

    @cached_property
    def explainer(self):
        return shap_explain.build_explainer(self.challenger, self.cfg)

    @cached_property
    def threshold(self) -> float:
        gov = json.loads((self.cfg.path("artifacts") / "governance_report.json").read_text())
        return float(gov["decision"]["threshold"])

    @cached_property
    def _vintage_rate(self) -> dict:
        """Per-vintage mean note rate (for the rate_spread feature) + global mean."""
        gold = pd.read_parquet(self.cfg.path("gold") / "feature_matrix.parquet")
        means = gold.groupby("vintage")["orig_interest_rate"].mean().to_dict()
        means["_global"] = float(gold["orig_interest_rate"].mean())
        return means

    # ---- feature engineering (mirrors pipeline.gold._engineer) ---------------
    def _build_features(self, raw: dict) -> pd.DataFrame:
        r = dict(raw)
        vint = r.get("vintage", "_global")
        vint_mean = self._vintage_rate.get(vint, self._vintage_rate["_global"])
        r["ltv_dti"] = round(r["ltv"] * r["dti"] / 100.0, 2)
        r["rate_spread"] = round(r["orig_interest_rate"] - vint_mean, 3)
        r["high_ltv"] = int(r["ltv"] > 80)
        row = {c: r[c] for c in feature_columns()}
        df = pd.DataFrame([row])
        for c in categorical_columns():
            df[c] = df[c].astype("category")
        for c in numeric_columns():
            df[c] = pd.to_numeric(df[c])
        return df

    def score(self, raw: dict) -> dict:
        X = self._build_features(raw)

        pd_sc = float(self.cal_sc.transform(self.scorecard.predict_proba(X)[:, 1])[0])
        pd_ch = float(self.cal_ch.transform(challenger.predict_pd(self.challenger, X))[0])
        credit_score = float(self.scorecard.score(X)[0])
        band = str(decision.assign_band(credit_score, self.cfg))

        lgd_hat = float(self.lgd.predict(X)[0])
        ead = float(ead_mod.estimate_ead(
            pd.DataFrame([{"orig_upb": raw["orig_upb"]}]), self.cfg)[0])
        el = float(el_mod.expected_loss(pd_ch, lgd_hat, ead))

        dec = str(decision.decide(pd_ch, self.threshold))
        local = shap_explain.local_attribution(self.explainer, X)
        codes = reason_codes.reason_codes(local, top_n=4)

        return {
            "pd_scorecard": pd_sc,
            "pd_challenger": pd_ch,
            "pd": pd_ch,
            "credit_score": round(credit_score, 1),
            "risk_band": band,
            "lgd": round(lgd_hat, 4),
            "ead": round(ead, 2),
            "expected_loss": round(el, 2),
            "decision": dec,
            "threshold": self.threshold,
            "reason_codes": codes if dec == "decline" else [],
            "explanation": [
                {"feature": t.feature, "value": str(t.value), "shap": round(float(t.shap), 4)}
                for t in local.itertuples(index=False)
            ],
            "adverse_action_letter": reason_codes.narrate(codes, dec),
        }
