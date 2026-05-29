"""Agent toolbelt — the functions the specialist agents may call.

Every tool wraps the *real* platform (the same ModelBundle, scored set, and
report artifacts the cockpit uses), returns a compact `summary` for the LLM, and
may attach a `_chart` spec the runtime renders. Numbers come from the models —
never from the language model.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from creditforge.agents import charts
from creditforge.config import load_config
from creditforge.governance import decision
from creditforge.serving import ModelBundle

_bundle: ModelBundle | None = None


def _b() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle


@lru_cache(maxsize=1)
def _scored() -> pd.DataFrame:
    """OOT scored set merged with Gold features, with band + decision attached."""
    cfg = load_config()
    s = pd.read_parquet(cfg.path("artifacts") / "test_scored.parquet")
    gold = pd.read_parquet(cfg.path("gold") / "feature_matrix.parquet")
    feat_cols = [c for c in ("fico", "dti", "ltv", "orig_interest_rate", "orig_upb",
                             "occupancy_status", "loan_purpose", "property_type",
                             "first_time_homebuyer") if c in gold.columns]
    s = s.merge(gold[["loan_id", *feat_cols]], on="loan_id", how="left")
    s["risk_band"] = decision.assign_band(s["credit_score"].to_numpy(), cfg)
    s["decision"] = decision.decide(s["pd_challenger"].to_numpy(), _b().threshold)
    return s


def _report(name: str) -> dict:
    return json.loads((load_config().path("artifacts") / name).read_text())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def score_borrower(fico: int, dti: float, ltv: int, orig_interest_rate: float,
                   orig_upb: int, orig_loan_term: int = 360,
                   occupancy_status: str = "P", loan_purpose: str = "P",
                   property_type: str = "SF", first_time_homebuyer: str = "N",
                   vintage: str | None = None) -> dict:
    """Score one borrower end-to-end (PD, score, band, LGD, EAD, Expected Loss,
    decision, adverse-action reasons)."""
    r = _b().score(dict(
        fico=fico, dti=dti, ltv=ltv, orig_interest_rate=orig_interest_rate,
        orig_upb=orig_upb, orig_loan_term=orig_loan_term,
        occupancy_status=occupancy_status, loan_purpose=loan_purpose,
        property_type=property_type, first_time_homebuyer=first_time_homebuyer,
        vintage=vintage or "2021-06"))
    expl = sorted(r["explanation"], key=lambda e: abs(e["shap"]), reverse=True)[:7]
    chart = charts.hbar(
        "Why this decision — local SHAP (right = raises PD)",
        [{"name": e["feature"], "value": e["shap"]} for e in expl],
        value_format="num", x_label="SHAP contribution", diverging=True)
    return {"summary": {
        "decision": r["decision"], "pd": round(r["pd"], 4),
        "credit_score": r["credit_score"], "risk_band": r["risk_band"],
        "lgd": r["lgd"], "ead": r["ead"], "expected_loss": r["expected_loss"],
        "reason_codes": [c["reason"] for c in r["reason_codes"]],
    }, "_chart": chart}


def portfolio_summary() -> dict:
    """Headline portfolio risk: loan count, Expected Loss, EL rate, approval
    rate, and the risk-band distribution."""
    s = _scored()
    order = [b["name"] for b in load_config().decision.bands]
    band = s["risk_band"].value_counts().reindex(order).fillna(0).astype(int)
    chart = charts.bar("Risk-band concentration", [{"band": k, "loans": int(v)}
                       for k, v in band.items()], x="band", y="loans",
                       label="loans", colors=[charts.BAND_COLORS.get(k) for k in band.index])
    return {"summary": {
        "n_loans": int(len(s)),
        "portfolio_expected_loss": round(float(s["expected_loss"].sum()), 0),
        "total_ead": round(float(s["ead"].sum()), 0),
        "el_rate_bps": round(float(s["expected_loss"].sum() / s["ead"].sum() * 1e4), 1),
        "mean_pd": round(float(s["pd_challenger"].mean()), 4),
        "approval_rate": round(float((s["decision"] == "approve").mean()), 4),
        "band_distribution": band.to_dict(),
    }, "_chart": chart}


def portfolio_slice(by: str, metric: str = "default_rate") -> dict:
    """Slice the book by a dimension and compute a risk metric per group.
    by: vintage | risk_band | borrower_race | occupancy_status | loan_purpose |
        fico_band | ltv_band
    metric: default_rate | mean_pd | expected_loss | count"""
    s = _scored().copy()
    if by == "fico_band":
        s["fico_band"] = pd.cut(s["fico"], [0, 640, 680, 720, 760, 850],
                                labels=["<640", "640-680", "680-720", "720-760", "760+"])
    elif by == "ltv_band":
        s["ltv_band"] = pd.cut(s["ltv"], [0, 70, 80, 90, 125],
                               labels=["<70", "70-80", "80-90", "90+"])
    if by not in s.columns:
        return {"summary": {"error": f"unknown dimension '{by}'"}}

    g = s.groupby(by, observed=True).agg(
        count=("loan_id", "size"),
        default_rate=("default_12m", "mean"),
        mean_pd=("pd_challenger", "mean"),
        expected_loss=("expected_loss", "sum")).reset_index()
    g = g[g["count"] >= 20] if by == "vintage" else g
    fmt = {"default_rate": "pct", "mean_pd": "pct",
           "expected_loss": "money", "count": "num"}[metric]
    rows = [{by: str(r[by]), metric: round(float(r[metric]), 4)} for _, r in g.iterrows()]
    chart = charts.bar(f"{metric.replace('_', ' ')} by {by}", rows, x=by, y=metric,
                       label=metric, value_format=fmt,
                       colors=[charts.BAND_COLORS.get(str(r[by])) for _, r in g.iterrows()]
                       if by == "risk_band" else None)
    table = g.assign(default_rate=g["default_rate"].round(4),
                     mean_pd=g["mean_pd"].round(4),
                     expected_loss=g["expected_loss"].round(0)).to_dict(orient="records")
    return {"summary": {"by": by, "metric": metric, "groups": table[:40]}, "_chart": chart}


def top_exposures(n: int = 10) -> dict:
    """The largest Expected-Loss exposures in the book."""
    s = _scored().sort_values("expected_loss", ascending=False).head(min(n, 25))
    cols = ["loan_id", "vintage", "pd_challenger", "credit_score", "risk_band",
            "lgd_hat", "ead", "expected_loss", "decision"]
    rows = s[cols].round({"pd_challenger": 4, "credit_score": 0, "lgd_hat": 4,
                          "ead": 0, "expected_loss": 0}).to_dict(orient="records")
    return {"summary": {"top_exposures": rows}}


def model_drivers() -> dict:
    """Global feature importance (mean |SHAP|) for the champion model."""
    g = _report("shap_global.json")[:8]
    chart = charts.hbar("Global drivers — mean |SHAP|",
                        [{"name": d["feature"], "value": round(d["mean_abs_shap"], 4)}
                         for d in g], value_format="num", x_label="mean |SHAP|")
    return {"summary": {"top_drivers": [
        {"feature": d["feature"], "mean_abs_shap": round(d["mean_abs_shap"], 4)} for d in g
    ]}, "_chart": chart}


def validation_metrics() -> dict:
    """Out-of-time discrimination, calibration, and the scorecard-vs-challenger
    benchmark, with pass/fail vs the gate thresholds."""
    v = _report("validation_report.json")
    t = v["thresholds"]
    out = {}
    for m in ("scorecard", "challenger"):
        d, c = v["models"][m]["discrimination"], v["models"][m]["calibration"]
        out[m] = {"gini": round(d["gini"], 4), "ks": round(d["ks"], 4),
                  "calibration_max_band_error": round(c["max_band_error"], 4),
                  "ece": round(c["ece"], 4)}
    chart = charts.bar("Gini — scorecard vs challenger (OOT)",
                       [{"model": "scorecard", "gini": out["scorecard"]["gini"]},
                        {"model": "challenger", "gini": out["challenger"]["gini"]}],
                       x="model", y="gini", label="Gini", value_format="num",
                       reference={"value": t["gini_min"], "label": "min gate"})
    return {"summary": {"metrics": out, "benchmark": {
        "gini_gap": round(v["benchmark"]["gini_gap"], 4),
        "verdict": v["benchmark"]["verdict"]},
        "thresholds": {"gini_min": t["gini_min"], "ks_min": t["ks_min"],
                       "max_band_error": t["calibration_max_band_error"]}}, "_chart": chart}


def validation_curves(which: str = "reliability", model: str = "challenger") -> dict:
    """Reliability (predicted vs observed) or cumulative gains curve, OOT."""
    v = _report("validation_report.json")
    m = v["models"].get(model, v["models"]["challenger"])
    if which == "gains":
        g = m["discrimination"]["gains"]
        data = [{"decile": int(r["band"]), "capture": round(r["cum_capture_rate"], 4)} for r in g]
        chart = charts.line("Cumulative gains (defaults captured)", data, x="decile",
                            series=[{"key": "capture", "label": "captured", "color": charts.GREEN}],
                            value_format="pct", x_label="risk decile")
    else:
        rc = m["calibration"]["reliability"]
        data = [{"x": round(r["predicted"], 4), "y": round(r["observed"], 4)} for r in rc]
        mx = max([d["x"] for d in data] + [d["y"] for d in data] + [0.05])
        chart = charts.scatter("Reliability — predicted vs observed PD", data,
                               value_format="pct", x_label="predicted", y_label="observed",
                               diagonal=True, domain=[0, round(mx, 3)])
    return {"summary": {"curve": which, "model": model, "points": len(chart["data"])},
            "_chart": chart}


def stability_metrics() -> dict:
    """Score PSI across vintages and the worst feature CSI (drift)."""
    v = _report("validation_report.json")
    st, t = v["stability"], v["thresholds"]
    psi = st["score_psi_by_vintage"]
    chart = charts.line("Score PSI across vintages", [
        {"vintage": r["vintage"], "psi": round(r["psi"], 4)} for r in psi],
        x="vintage", series=[{"key": "psi", "label": "PSI", "color": charts.ACCENT}],
        value_format="num", reference={"value": t["psi_unstable"], "label": "unstable"})
    return {"summary": {
        "max_score_psi": round(st["max_score_psi"], 4),
        "max_feature_csi": round(st["max_feature_csi"], 4),
        "psi_watch": t["psi_watch"], "psi_unstable": t["psi_unstable"],
        "status": "stable" if st["max_score_psi"] < t["psi_watch"] else "watch/unstable",
    }, "_chart": chart}


def fairness_metrics() -> dict:
    """Disparate-impact and equal-opportunity by protected group at the decision
    threshold (the model never uses race; this measures proxy effects)."""
    gov = _report("governance_report.json")
    f = gov["fairness"]
    groups = sorted(f["groups"], key=lambda g: g["disparate_impact"], reverse=True)
    chart = charts.bar(f"Disparate impact by group (vs {f['privileged_group']})",
                       [{"group": g["group"], "disparate_impact": round(g["disparate_impact"], 3)}
                        for g in groups], x="group", y="disparate_impact",
                       label="DI ratio", value_format="num",
                       colors=[charts.GREEN if g["disparate_impact"] >= f["di_ratio_floor"]
                               else charts.RED for g in groups],
                       reference={"value": f["di_ratio_floor"], "label": "4/5ths floor"})
    return {"summary": {
        "min_disparate_impact": round(f["di_ratio_min_observed"], 3),
        "four_fifths_floor": f["di_ratio_floor"],
        "passes": f["passes_four_fifths"],
        "max_equal_opportunity_diff": round(f["max_equal_opportunity_diff"], 3),
        "groups": [{"group": g["group"], "approval_rate": round(g["approval_rate"], 3),
                    "default_rate": round(g["default_rate"], 3),
                    "disparate_impact": round(g["disparate_impact"], 3),
                    "equal_opportunity_diff": round(g["equal_opportunity_diff"], 3)}
                   for g in groups],
    }, "_chart": chart}


# ---------------------------------------------------------------------------
# Registry + JSON schemas for function-calling
# ---------------------------------------------------------------------------
_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}


def _schema(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": required or []}}}


TOOLS = {
    "score_borrower": (score_borrower, _schema(
        "score_borrower",
        "Score a single mortgage applicant: PD, credit score, risk band, LGD, EAD, "
        "Expected Loss, approve/decline decision, and adverse-action reason codes.",
        {"fico": _INT, "dti": _NUM, "ltv": _INT, "orig_interest_rate": _NUM,
         "orig_upb": _INT, "orig_loan_term": _INT,
         "occupancy_status": {"type": "string", "enum": ["P", "I", "S"]},
         "loan_purpose": {"type": "string", "enum": ["P", "C", "N"]},
         "property_type": {"type": "string", "enum": ["SF", "CO", "PU", "MH"]},
         "first_time_homebuyer": {"type": "string", "enum": ["Y", "N"]},
         "vintage": _STR},
        ["fico", "dti", "ltv", "orig_interest_rate", "orig_upb"])),
    "portfolio_summary": (portfolio_summary, _schema(
        "portfolio_summary", "Headline portfolio risk: loans, Expected Loss, EL "
        "rate (bps), approval rate, and risk-band distribution.")),
    "portfolio_slice": (portfolio_slice, _schema(
        "portfolio_slice", "Group the loan book by a dimension and compute a risk "
        "metric per group (returns a bar chart).",
        {"by": {"type": "string", "enum": ["vintage", "risk_band", "borrower_race",
                "occupancy_status", "loan_purpose", "fico_band", "ltv_band"]},
         "metric": {"type": "string", "enum": ["default_rate", "mean_pd",
                    "expected_loss", "count"]}}, ["by"])),
    "top_exposures": (top_exposures, _schema(
        "top_exposures", "List the largest Expected-Loss exposures in the book.",
        {"n": _INT})),
    "model_drivers": (model_drivers, _schema(
        "model_drivers", "Global feature importance (mean |SHAP|) for the champion model.")),
    "validation_metrics": (validation_metrics, _schema(
        "validation_metrics", "Out-of-time Gini/KS/calibration and the scorecard-"
        "vs-challenger benchmark, with gate thresholds.")),
    "validation_curves": (validation_curves, _schema(
        "validation_curves", "Reliability curve or cumulative-gains curve (OOT).",
        {"which": {"type": "string", "enum": ["reliability", "gains"]},
         "model": {"type": "string", "enum": ["scorecard", "challenger"]}})),
    "stability_metrics": (stability_metrics, _schema(
        "stability_metrics", "Score PSI across vintages + worst feature CSI (drift).")),
    "fairness_metrics": (fairness_metrics, _schema(
        "fairness_metrics", "Disparate impact and equal-opportunity by protected "
        "group at the decision threshold, vs the 4/5ths rule.")),
}


def schemas_for(names: list[str]) -> list[dict]:
    return [TOOLS[n][1] for n in names if n in TOOLS]


def call_tool(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"summary": {"error": f"unknown tool {name}"}}
    try:
        return TOOLS[name][0](**args)
    except Exception as e:  # surface tool errors to the agent instead of crashing
        return {"summary": {"error": f"{type(e).__name__}: {e}"}}
