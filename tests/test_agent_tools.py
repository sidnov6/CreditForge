"""Agent toolbelt tests — deterministic, no LLM needed.

Verifies the tools return well-formed summaries + valid chart specs, so the agent
layer is testable in CI without a live model key. Skips cleanly if artifacts
haven't been generated yet.
"""
import pytest

from creditforge.config import load_config

pytestmark = pytest.mark.skipif(
    not (load_config().path("artifacts") / "test_scored.parquet").exists(),
    reason="run the pipeline first (python -m creditforge.cli all)")

from creditforge.agents import tools  # noqa: E402

VALID_CHART_TYPES = {"bar", "hbar", "line", "scatter"}


def _check_chart(c):
    assert c["type"] in VALID_CHART_TYPES
    assert c["title"] and c["data"]
    assert "id" in c


def test_portfolio_summary():
    r = tools.portfolio_summary()
    s = r["summary"]
    assert s["n_loans"] > 0 and 0 <= s["approval_rate"] <= 1
    assert set(s["band_distribution"]) <= {"AAA", "AA", "A", "BBB", "BB", "B"}
    _check_chart(r["_chart"])


def test_portfolio_slice_fico_monotonic_ish():
    r = tools.portfolio_slice(by="fico_band", metric="default_rate")
    groups = r["summary"]["groups"]
    assert len(groups) >= 3
    _check_chart(r["_chart"])


def test_score_borrower_high_risk_declines():
    r = tools.score_borrower(fico=640, dti=48, ltv=95, orig_interest_rate=6.0,
                             orig_upb=350000, occupancy_status="I", loan_purpose="C",
                             property_type="MH", first_time_homebuyer="Y")
    s = r["summary"]
    assert s["decision"] in ("approve", "decline")
    assert 0 <= s["pd"] <= 1 and s["reason_codes"]
    assert r["_chart"]["type"] == "hbar"


def test_fairness_metrics_shape():
    s = tools.fairness_metrics()["summary"]
    assert "min_disparate_impact" in s and isinstance(s["passes"], bool)
    assert len(s["groups"]) >= 2


def test_validation_metrics_and_curves():
    s = tools.validation_metrics()["summary"]
    assert "challenger" in s["metrics"] and "gini" in s["metrics"]["challenger"]
    _check_chart(tools.validation_curves(which="reliability")["_chart"])
    _check_chart(tools.validation_curves(which="gains")["_chart"])


def test_schemas_cover_all_tools():
    names = list(tools.TOOLS)
    schemas = tools.schemas_for(names)
    assert len(schemas) == len(names)
    for sc in schemas:
        assert sc["type"] == "function" and "name" in sc["function"]


def test_unknown_tool_is_safe():
    out = tools.call_tool("does_not_exist", {})
    assert "error" in out["summary"]
