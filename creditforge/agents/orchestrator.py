"""Orchestrator — routes a question to specialist agents, then synthesizes.

Flow: a routing call picks 1+ specialists (capped) and a sub-task for each ->
each specialist runs its own tool loop independently -> a synthesis call fuses
their findings into one answer. Charts from every specialist are merged and
returned for the cockpit to render.
"""
from __future__ import annotations

import json

from creditforge.agents import llm
from creditforge.agents.specialists import SPECIALISTS, run_specialist
from creditforge.config import load_config

_ROUTER_SYSTEM = (
    "You are the orchestrator of a credit-risk analysis team. Route the user's "
    "question to the right specialists and give each a precise sub-task.\n"
    "Specialists:\n"
    "- analyst: portfolio risk, Expected Loss, segments/concentrations, scoring a "
    "borrower, what drives risk.\n"
    "- validator: model performance — Gini/KS, calibration, stability/PSI, "
    "scorecard-vs-challenger, does it pass.\n"
    "- fairness: disparate impact / equal opportunity by protected group, mitigation.\n"
    "Reply with ONLY a JSON object: "
    '{"plan":[{"specialist":"analyst","task":"..."}]}. '
    "Use 1 specialist for focused questions, up to the cap for cross-cutting ones. "
    "For a greeting or a question unrelated to the platform, return "
    '{"plan":[], "direct":"<short reply>"}.')


def _route(question: str, max_specialists: int) -> dict:
    resp = llm.complete(
        [{"role": "system", "content": _ROUTER_SYSTEM},
         {"role": "user", "content": question}],
        temperature=0.0)
    raw = resp.choices[0].message.content or "{}"
    raw = raw[raw.find("{"): raw.rfind("}") + 1] or "{}"
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {"plan": [{"specialist": "analyst", "task": question}]}
    valid = [p for p in plan.get("plan", []) if p.get("specialist") in SPECIALISTS]
    plan["plan"] = valid[:max_specialists]
    return plan


def _compose(findings: list[dict]) -> str:
    """Combine specialists' tool-grounded answers under headings.

    We deliberately do NOT re-summarize through the LLM: a second pass over the
    numbers is where weaker free-tier models hallucinate figures. Each section is
    the specialist's own answer, which is grounded in a forced tool call.
    """
    return "\n\n".join(f"**{f['title']}**\n\n{f['text']}" for f in findings)


def run(question: str) -> dict:
    cfg = load_config()
    plan = _route(question, int(cfg.agents.max_specialists))

    if not plan["plan"]:
        return {"answer": plan.get("direct", "I focus on this credit-risk platform — "
                "ask about the portfolio, model validation, fairness, or scoring a "
                "borrower."), "charts": [], "trace": [], "agents": []}

    findings, charts, trace, agents = [], [], [], []
    for step in plan["plan"]:
        out = run_specialist(step["specialist"], step["task"])
        findings.append(out)
        charts.extend(out["charts"])
        agents.append({"specialist": out["specialist"], "title": out["title"]})
        trace.extend({"agent": out["title"], **t} for t in out["trace"])

    answer = findings[0]["text"] if len(findings) == 1 else _compose(findings)
    return {"answer": answer, "charts": charts, "trace": trace, "agents": agents}
