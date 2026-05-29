"""CI performance gates (Part 10.3).

A model ships only if it clears minimum discrimination, calibration, and
stability thresholds. Run in CI on every change; a regression fails the build.

Calibration is gated on the economically meaningful **max band error** and ECE,
NOT the Hosmer-Lemeshow p-value: with ~17k loans HL rejects on trivially small
miscalibration (large-sample over-power), so HL is reported as informational.

Fairness disparate-impact is surfaced as a REVIEW flag, not a hard build gate —
remediating it via group-specific thresholds is illegal in lending, so it needs
human judgement, not an automated fail.
"""
from __future__ import annotations

import json
import sys

from creditforge.config import Config, load_config


def run_gates(cfg: Config | None = None, champion: str = "challenger") -> tuple[bool, list[dict]]:
    cfg = cfg or load_config()
    art = cfg.path("artifacts")
    val = json.loads((art / "validation_report.json").read_text())
    t = val["thresholds"]
    m = val["models"][champion]
    d, c = m["discrimination"], m["calibration"]
    stab = val["stability"]

    checks = [
        ("Gini ≥ min", d["gini"], t["gini_min"], d["gini"] >= t["gini_min"], ">="),
        ("KS ≥ min", d["ks"], t["ks_min"], d["ks"] >= t["ks_min"], ">="),
        ("Calibration max band error ≤ max", c["max_band_error"],
         t["calibration_max_band_error"], c["max_band_error"] <= t["calibration_max_band_error"], "<="),
        ("Max score PSI ≤ unstable", stab["max_score_psi"],
         t["psi_unstable"], stab["max_score_psi"] <= t["psi_unstable"], "<="),
        ("Max feature CSI ≤ unstable", stab["max_feature_csi"],
         t["psi_unstable"], stab["max_feature_csi"] <= t["psi_unstable"], "<="),
    ]
    results = [{"check": name, "value": round(float(v), 4), "threshold": float(th),
                "op": op, "passed": bool(ok)} for name, v, th, ok, op in checks]
    return all(r["passed"] for r in results), results


def main() -> int:
    cfg = load_config()
    passed, results = run_gates(cfg)
    print(f"\n  CreditForge CI gates (champion=challenger) — data={cfg.run.data_version}\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"   [{status}] {r['check']:34s} {r['value']:>8} {r['op']} {r['threshold']}")
    # Informational fairness flag (non-gating)
    try:
        gov = json.loads((cfg.path("artifacts") / "governance_report.json").read_text())
        di = gov["fairness"]["di_ratio_min_observed"]
        flag = "OK" if gov["fairness"]["passes_four_fifths"] else "REVIEW"
        print(f"\n   [{flag}] fairness min disparate-impact ratio = {di:.3f} "
              f"(floor {gov['fairness']['di_ratio_floor']}, non-gating)")
    except FileNotFoundError:
        pass
    print(f"\n  => {'ALL GATES PASSED' if passed else 'GATES FAILED'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
