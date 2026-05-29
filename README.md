# CreditForge · `BONITAS`

**🌐 Live demo: https://huggingface.co/spaces/sidnov6/CreditForge**

**Bank-grade credit-risk modeling & validation platform.** A calibrated PD model
(regulatory WoE scorecard *and* an ML challenger), LGD, and EAD combined into
Expected Loss — wrapped in a bank-grade validation suite, SHAP explainability,
fairness testing, and drift monitoring. Built to the global Basel/IRB methodology
on mortgage-performance data.

```
EL = PD × LGD × EAD
```

> Methodology is jurisdiction-agnostic (Basel/IRB). The pipeline reads the
> **Freddie Mac Single-Family** loan + monthly-performance schema; until you drop
> real GSE files in, a **seeded synthetic generator** produces the same schema so
> the whole stack runs offline and reproducibly.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS LightGBM: brew install libomp
python -m creditforge.cli generate       # synthetic GSE panels -> Bronze
python -m creditforge.cli pipeline       # Bronze -> Silver -> Gold (+ OOT split)
python -m creditforge.cli train          # scorecard + challenger + calibration
python -m creditforge.cli validate       # Gini/KS/calibration/PSI -> report
python -m creditforge.cli all            # the whole lifecycle end-to-end
```

## Deploy (single container)

The whole app ships as one Docker image: FastAPI serves the API under `/api` and
the pre-built static Next.js cockpit at `/` (one origin, no CORS). Models are
trained and baked at build time, so the runtime image only serves.

```bash
docker build -t creditforge .
docker run -p 7860:7860 creditforge        # open http://localhost:7860
```

Deployed free on Hugging Face Spaces (Docker SDK) — see
[`deploy/push_to_hf.py`](deploy/push_to_hf.py). Live: the link at the top.

## Architecture (lifecycle, not an agent graph)

```
Bronze (raw, vintage-partitioned)
  -> Silver (cleaned, point-in-time 12m default flag, perf joined)
    -> Gold (leakage-safe feature matrix + target, vintage-tagged)
      -> OOT split (train: old vintages / test: new vintages)
        -> PD scorecard (WoE/IV + logistic)  ─┐
        -> PD challenger (LightGBM)           ─┼─> calibration (isotonic)
                                               │
        -> LGD (two-stage) -> EAD/CCF ─────────┴─> Expected Loss
          -> Validation (Gini/KS/gains, calibration/HL, PSI/CSI, benchmark)
            -> Governance (SHAP, reason codes, fairness, model card)
              -> Serving (FastAPI) + Risk Cockpit (Next.js) + drift monitoring
```

## Why this is bank-credible, not a Kaggle notebook
The model is ~20% of the work. The 80% that proves maturity is the scaffolding:
**leakage-safe point-in-time target, out-of-time validation**, WoE scorecards,
calibration, Gini/KS/PSI, adverse-action reason codes, fairness, a model card,
CI performance gates, and drift monitoring.

## Layout
```
creditforge/
  conf/config.yaml      # all thresholds/scaling/cutoffs — no hardcoding
  pipeline/             # synthetic generator + Bronze/Silver/Gold + split
  models/               # scorecard, challenger, calibration, lgd, ead, el
  validation/           # discrimination, calibration_tests, stability, report
  governance/           # shap_explain, reason_codes, fairness, model_card
  monitoring/           # scheduled PSI drift
  eval/                 # CI performance gates (Gini/cal/PSI thresholds)
app/api/                # FastAPI scoring service
app/dashboard/          # Next.js Risk Cockpit (5 screens)
tests/                  # transform + EL-math unit tests, leakage tripwires
```

The leakage discipline is the keystone: features use only information available
at the observation date, the target looks forward 12 months, and the split is
out-of-time — never random. That is what separates a credible credit model from
a leaky one.
