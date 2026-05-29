"""Governance aggregator — one report the model-governance function expects.

Pulls together the decision policy, fairness analysis, global SHAP drivers, and
a worked adverse-action example, then emits:
  - artifacts/governance_report.json  (dashboard Governance screen)
  - governance/model_card.md          (the one-page model card artifact)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from creditforge.config import Config, load_config, run_stamp
from creditforge.dataset import feature_columns
from creditforge.governance import decision, fairness, reason_codes, shap_explain
from creditforge.pipeline.split import make_split

CHAMPION_PD = "pd_challenger"


def build_governance(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    art = cfg.path("artifacts")
    scored = pd.read_parquet(art / "test_scored.parquet")
    train_metrics = json.loads((art / "train_metrics.json").read_text())
    validation = json.loads((art / "validation_report.json").read_text())

    thr = decision.default_threshold(scored[CHAMPION_PD], scored["default_12m"], cfg)
    dec_summary = decision.decision_summary(scored, CHAMPION_PD, thr, cfg)
    fair = fairness.fairness_report(scored, CHAMPION_PD, thr, cfg)

    # Global SHAP drivers (precomputed artifact, refresh if missing)
    shap_path = art / "shap_global.json"
    if not shap_path.exists():
        shap_explain.save_global(cfg)
    shap_global = json.loads(shap_path.read_text())

    # Worked adverse-action example: a representative declined applicant
    test = make_split(cfg=cfg).test
    test = test.merge(scored[["loan_id", CHAMPION_PD, "credit_score"]], on="loan_id")
    declined = test[test[CHAMPION_PD] > thr].sort_values(CHAMPION_PD, ascending=False)
    example = None
    if len(declined):
        row = declined.iloc[[len(declined) // 2]]   # a median-risk decline
        expl = shap_explain.build_explainer(cfg=cfg)
        local = shap_explain.local_attribution(expl, row[feature_columns()])
        codes = reason_codes.reason_codes(local, top_n=4)
        example = {
            "loan_id": int(row["loan_id"].iloc[0]),
            "pd": float(row[CHAMPION_PD].iloc[0]),
            "credit_score": float(row["credit_score"].iloc[0]),
            "band": decision.assign_band(row["credit_score"].iloc[0], cfg),
            "decision": "decline",
            "reason_codes": codes,
            "letter": reason_codes.narrate(codes, "decline"),
        }

    report = {
        "stamp": run_stamp(),
        "champion_model": "challenger (LightGBM, isotonic-calibrated)",
        "decision": dec_summary,
        "fairness": fair,
        "shap_global": shap_global,
        "adverse_action_example": example,
    }
    (art / "governance_report.json").write_text(json.dumps(report, indent=2, default=float))
    _write_model_card(report, train_metrics, validation, cfg)
    return report


def _write_model_card(gov: dict, tm: dict, val: dict, cfg: Config) -> None:
    b = val["benchmark"]
    el = tm["expected_loss"]
    fair = gov["fairness"]
    di = fair["di_ratio_min_observed"]
    sc_gini = val["models"]["scorecard"]["discrimination"]["gini"]
    ch_gini = val["models"]["challenger"]["discrimination"]["gini"]
    ch_cal = val["models"]["challenger"]["calibration"]
    top_drivers = ", ".join(d["feature"] for d in gov["shap_global"][:5])
    s = gov["stamp"]

    card = f"""# Model Card — CreditForge PD/LGD/EAD Expected-Loss Model

> Auto-generated. data_version=`{s['data_version']}` · code_version=`{s['code_version']}` · seed=`{s['seed']}`

## 1. Model details
- **Purpose:** Estimate 12-month Probability of Default (PD), Loss Given Default
  (LGD), Exposure at Default (EAD) and combine them into Expected Loss
  (EL = PD × LGD × EAD) for a residential-mortgage portfolio.
- **Models:** (A) regulatory **WoE/IV logistic scorecard**; (B) **LightGBM
  challenger**, isotonic-calibrated. Champion for EL/decisioning: **{gov['champion_model']}**.
- **Methodology:** Global Basel/IRB framework (jurisdiction-agnostic). Data is the
  Freddie Mac Single-Family schema (synthetic generator standing in for the GSE
  files; real files drop into the same pipeline).

## 2. Intended use
- Portfolio Expected-Loss estimation, IFRS-9-style provisioning, and lending
  decision support, with human oversight. **Not** for automated decisions without
  review, and not validated outside US-style residential mortgages.

## 3. Data & leakage discipline
- Point-in-time features (origination only), forward-looking 12-month target,
  **out-of-time** split (train ≤ {cfg.target.oot.train_vintage_end}, test ≥
  {cfg.target.oot.test_vintage_start}). No performance data leaks into features.
- Protected attribute (`borrower_race`) is **never** a model input.

## 4. Performance (out-of-time)
| Metric | Scorecard | Challenger |
|---|---|---|
| Gini | {sc_gini:.4f} | {ch_gini:.4f} |
| KS | {val['models']['scorecard']['discrimination']['ks']:.4f} | {val['models']['challenger']['discrimination']['ks']:.4f} |
| Calibration max band error | {val['models']['scorecard']['calibration']['max_band_error']:.4f} | {ch_cal['max_band_error']:.4f} |
| ECE | {val['models']['scorecard']['calibration']['ece']:.4f} | {ch_cal['ece']:.4f} |

- **Benchmark:** challenger − scorecard Gini gap = {b['gini_gap']:+.4f}. {b['verdict']}
- **Top global drivers (SHAP):** {top_drivers}.
- **Portfolio:** EL ≈ {el['portfolio_el']:,.0f} on EAD {el['total_ead']:,.0f}
  (≈ {el['el_rate_bps']:.0f} bps); mean PD {el['mean_pd']:.2%}, mean LGD {el['mean_lgd_hat']:.2%}.

## 5. Fairness
- Disparate-impact (4/5ths) at the decision threshold: min ratio **{di:.3f}**
  (floor {fair['di_ratio_floor']}) → **{'PASS' if fair['passes_four_fifths'] else 'REVIEW'}**.
- Max equal-opportunity difference: {fair['max_equal_opportunity_diff']:.3f}.
- **Discussion:** the model excludes race, yet neutral features (e.g. credit
  score, DTI) proxy historical disparity, producing disparate impact. Mitigation
  options: feature review, reject-option / fairness-constrained training
  (fairlearn), and monitoring — *not* group-specific thresholds (illegal in
  lending). Surfacing and tracking this is a regulatory expectation (EU AI Act
  high-risk credit scoring).

## 6. Limitations
- Synthetic data calibrated to plausible mortgage behaviour, not a specific
  institution's book. Macro scenarios are implicit in vintages, not modelled
  explicitly. LGD trained on a modest defaulted population.

## 7. Monitoring & governance
- CI performance gates (Gini / calibration / PSI). Scheduled PSI/CSI drift
  monitoring vs the training baseline. MLflow tracking + model registry for
  reproducibility. Every artifact stamped with code+data version and seed.
"""
    out = Path(__file__).resolve().parent / "model_card.md"
    out.write_text(card)


if __name__ == "__main__":
    r = build_governance()
    f = r["fairness"]
    print(f"[governance] threshold={r['decision']['threshold']:.4f} "
          f"approval={r['decision']['approval_rate']:.1%} | "
          f"min DI={f['di_ratio_min_observed']:.3f} "
          f"-> {'PASS' if f['passes_four_fifths'] else 'REVIEW'}")
    print(f"[governance] model card -> creditforge/governance/model_card.md")
    if r["adverse_action_example"]:
        print("[governance] sample adverse-action reasons:")
        for c in r["adverse_action_example"]["reason_codes"]:
            print(f"    {c['rank']}. {c['reason']}")
