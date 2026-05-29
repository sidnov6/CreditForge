"""The specialist agents — each a focused role with its own slice of the toolbelt."""
from __future__ import annotations

from creditforge.agents import llm

_COMMON = ("You are part of CreditForge, a bank-grade credit-risk platform. ALWAYS "
           "call your tools to get the real numbers — the data is already loaded, so "
           "NEVER say you 'need data' or 'don't have access'; just call the tool. "
           "Never invent figures. Be concise and quantitative. When a tool returns a "
           "chart it is shown to the user automatically, so reference it ('see the "
           "chart') rather than listing every value. Use bank terminology (PD, LGD, "
           "EAD, Expected Loss, Gini, PSI, disparate impact).")

SPECIALISTS = {
    "analyst": {
        "title": "Portfolio Analyst",
        "tools": ["portfolio_summary", "portfolio_slice", "top_exposures",
                  "score_borrower", "model_drivers"],
        "system": _COMMON + " You are the PORTFOLIO ANALYST. You explain the book's "
        "risk, Expected Loss, concentrations and segments, score individual "
        "borrowers, and identify what drives risk. Prefer portfolio_slice to show "
        "differences across segments.",
    },
    "validator": {
        "title": "Model Validator",
        "tools": ["validation_metrics", "validation_curves", "stability_metrics",
                  "model_drivers"],
        "system": _COMMON + " You are the MODEL VALIDATOR (model-risk function). You "
        "judge discrimination (Gini/KS), calibration, stability (PSI/CSI) and the "
        "scorecard-vs-challenger trade-off against the gate thresholds, out-of-time. "
        "State plainly whether the model passes and why.",
    },
    "fairness": {
        "title": "Fairness Officer",
        "tools": ["fairness_metrics", "score_borrower"],
        "system": _COMMON + " You are the FAIRNESS OFFICER. You assess disparate "
        "impact (4/5ths rule) and equal-opportunity differences by protected group. "
        "Crucially: the model never uses race; disparity arises because neutral "
        "features proxy historical gaps. Discuss lawful mitigation (feature review, "
        "fairness-constrained training, monitoring) — never group-specific cutoffs.",
    },
}


def run_specialist(name: str, task: str) -> dict:
    spec = SPECIALISTS.get(name)
    if not spec:
        return {"text": f"(no specialist '{name}')", "trace": [], "charts": []}
    out = llm.run_tool_loop(spec["system"], task, spec["tools"])
    out["specialist"] = name
    out["title"] = spec["title"]
    return out
