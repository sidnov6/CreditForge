"""Adverse-action reason codes (Part 8).

Regulators (ECOA/Reg B in the US; comparable transparency duties under the EU AI
Act's high-risk classification of credit scoring) require that a declined
applicant be told *why*. We convert the top risk-increasing local SHAP
contributions into plain-language reasons. The reasons come from SHAP, not from
a language model — an optional LLM layer may only narrate them into a letter.
"""
from __future__ import annotations

import pandas as pd

# Human-readable templates per feature; {value} is the applicant's value.
REASON_TEMPLATES = {
    "fico": "Credit score is low ({value}) relative to lower-risk borrowers",
    "dti": "Debt-to-income ratio is elevated ({value}%)",
    "ltv": "Loan-to-value ratio is high ({value}%), leaving thin equity",
    "ltv_dti": "Combined leverage (high LTV and DTI together) is elevated",
    "high_ltv": "Loan-to-value exceeds the 80% threshold",
    "orig_interest_rate": "Note rate ({value}%) reflects higher pricing risk",
    "rate_spread": "Rate is priced above comparable loans of this vintage",
    "orig_upb": "Loan amount ({value}) is large relative to the book",
    "orig_loan_term": "Loan term ({value} months) carries higher term risk",
    "occupancy_status": "Occupancy type ({value}) carries higher risk (e.g. investment)",
    "loan_purpose": "Loan purpose ({value}) carries higher risk (e.g. cash-out)",
    "property_type": "Property type ({value}) carries higher risk",
    "first_time_homebuyer": "First-time-homebuyer status increases risk",
}

# Friendly category decodes for the templates
_DECODE = {
    "occupancy_status": {"P": "primary residence", "I": "investment", "S": "second home"},
    "loan_purpose": {"P": "purchase", "C": "cash-out refinance", "N": "rate/term refinance"},
    "property_type": {"SF": "single-family", "CO": "condo", "PU": "PUD", "MH": "manufactured"},
    "first_time_homebuyer": {"Y": "yes", "N": "no"},
}


def reason_codes(local_attr: pd.DataFrame, top_n: int = 4) -> list[dict]:
    """Top-N risk-INCREASING reasons (positive SHAP) for a decline, ranked."""
    increasing = local_attr[local_attr["shap"] > 0].head(top_n)
    out = []
    for rank, row in enumerate(increasing.itertuples(index=False), start=1):
        feat, val = row.feature, row.value
        val = _DECODE.get(feat, {}).get(val, val)
        template = REASON_TEMPLATES.get(feat, f"{feat} contributes to elevated risk")
        out.append({
            "rank": rank,
            "feature": feat,
            "reason": template.format(value=val),
            "contribution": round(float(row.shap), 4),
        })
    return out


def narrate(reasons: list[dict], decision: str, lang: str = "en") -> str:
    """Plain-language adverse-action summary (deterministic; no LLM required)."""
    if decision == "approve":
        return "Application approved. No adverse-action reasons apply."
    lead = ("Your application could not be approved at this time. The principal "
            "factors in this decision were:")
    bullets = "\n".join(f"  {r['rank']}. {r['reason']}" for r in reasons)
    return f"{lead}\n{bullets}"
