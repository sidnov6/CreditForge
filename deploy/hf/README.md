---
title: CreditForge Risk Cockpit
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CreditForge · BONITAS — Risk Cockpit

Bank-grade credit-risk modeling & validation platform. A calibrated PD model
(WoE/logistic regulatory **scorecard** + **LightGBM challenger**), **LGD**, and
**EAD** combined into **Expected Loss**, wrapped in an out-of-time validation
suite, SHAP-based adverse-action reason codes, fairness testing, and PSI drift
monitoring. FastAPI serves the API under `/api`; the static Next.js cockpit is
served at `/`.

Built to the global Basel/IRB methodology on a synthetic Freddie Mac-schema
generator (real GSE files drop into the same pipeline). Source:
https://github.com/sidnov6/CreditForge
