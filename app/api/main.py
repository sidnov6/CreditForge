"""CreditForge scoring API (FastAPI).

Serves single-applicant scoring (PD/LGD/EL + decision + reason codes) and the
precomputed report artifacts (validation, governance, monitoring, portfolio) so
the Next.js Risk Cockpit has one backend. Run:

    uvicorn app.api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from creditforge.config import load_config
from creditforge.governance import decision
from creditforge.serving import ModelBundle

app = FastAPI(title="CreditForge Scoring API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def bundle() -> ModelBundle:
    return ModelBundle()


def _artifact(name: str) -> dict:
    cfg = load_config()
    path = cfg.path("artifacts") / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found — run the pipeline first")
    return json.loads(path.read_text())


class Applicant(BaseModel):
    fico: int = Field(ge=300, le=850, examples=[680])
    dti: float = Field(ge=0, le=70, examples=[42.0])
    ltv: int = Field(ge=1, le=125, examples=[88])
    orig_interest_rate: float = Field(ge=0, le=25, examples=[4.5])
    orig_upb: int = Field(ge=1000, examples=[280000])
    orig_loan_term: int = Field(ge=60, le=480, examples=[360])
    occupancy_status: str = Field(examples=["I"])
    loan_purpose: str = Field(examples=["C"])
    property_type: str = Field(examples=["SF"])
    first_time_homebuyer: str = Field(examples=["N"])
    vintage: str | None = Field(default=None, examples=["2021-06"])


@app.get("/health")
def health():
    return {"status": "ok", "stamp": load_config()["run"]}


@app.post("/score")
def score(applicant: Applicant):
    try:
        return bundle().score(applicant.model_dump())
    except KeyError as e:
        raise HTTPException(422, f"missing feature: {e}")


@app.get("/validation")
def validation():
    return _artifact("validation_report.json")


@app.get("/governance")
def governance():
    return _artifact("governance_report.json")


@app.get("/monitoring")
def monitoring():
    return _artifact("monitoring_report.json")


@app.get("/shap/global")
def shap_global():
    return _artifact("shap_global.json")


@app.get("/portfolio")
def portfolio(n: int = 500):
    """Portfolio risk view: aggregates + a sample of scored loans for the UI."""
    cfg = load_config()
    scored = pd.read_parquet(cfg.path("artifacts") / "test_scored.parquet")
    thr = bundle().threshold
    bands = decision.assign_band(scored["credit_score"].to_numpy(), cfg)
    scored = scored.assign(risk_band=bands,
                           decision=decision.decide(scored["pd_challenger"].to_numpy(), thr))
    band_order = [b["name"] for b in cfg.decision.bands]
    band_dist = (scored["risk_band"].value_counts().reindex(band_order).fillna(0)
                 .astype(int).to_dict())
    el_by_band = (scored.groupby("risk_band")["expected_loss"].sum()
                  .reindex(band_order).fillna(0).round(2).to_dict())
    sample_cols = ["loan_id", "vintage", "pd_challenger", "credit_score",
                   "risk_band", "lgd_hat", "ead", "expected_loss", "decision",
                   "default_12m"]
    return {
        "n_loans": int(len(scored)),
        "portfolio_el": float(scored["expected_loss"].sum()),
        "total_ead": float(scored["ead"].sum()),
        "el_rate_bps": float(scored["expected_loss"].sum() / scored["ead"].sum() * 1e4),
        "mean_pd": float(scored["pd_challenger"].mean()),
        "approval_rate": float((scored["decision"] == "approve").mean()),
        "band_distribution": band_dist,
        "el_by_band": el_by_band,
        "pd_histogram": _histogram(scored["pd_challenger"], 25),
        "sample": scored.sort_values("expected_loss", ascending=False)
                        .head(n)[sample_cols].to_dict(orient="records"),
    }


def _histogram(series: pd.Series, bins: int) -> list[dict]:
    counts, edges = pd.cut(series, bins=bins, retbins=True)
    vc = counts.value_counts(sort=False)
    return [{"x": round(float(edges[i]), 4), "count": int(vc.iloc[i])}
            for i in range(len(vc))]
