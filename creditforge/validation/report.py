"""Validation report — the bank-credible centerpiece (Part 7).

Runs the full suite on the out-of-time scored set for BOTH PD models, quantifies
the scorecard-vs-challenger gap, measures score stability across vintages and
feature stability train-vs-test, and emits:
  - artifacts/validation_report.json  (consumed by dashboard + CI gates)
  - reports/validation_report.html    (standalone portfolio artifact)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from creditforge.config import Config, load_config, run_stamp
from creditforge.dataset import categorical_columns, numeric_columns
from creditforge.pipeline.split import make_split
from creditforge.validation import calibration_tests as cal
from creditforge.validation import discrimination as disc
from creditforge.validation import stability as stab

MODELS = {"scorecard": "pd_scorecard", "challenger": "pd_challenger"}


def build_report(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    v = cfg.validation
    scored = pd.read_parquet(cfg.path("artifacts") / "test_scored.parquet")
    y = scored["default_12m"]

    # ---- per-model discrimination + calibration -------------------------------
    models = {}
    for name, col in MODELS.items():
        models[name] = {
            "discrimination": disc.discrimination_report(y, scored[col], int(v.n_score_bands)),
            "calibration": cal.calibration_report(y, scored[col], int(v.n_score_bands)),
        }

    # ---- benchmark: scorecard vs challenger -----------------------------------
    g_sc = models["scorecard"]["discrimination"]["gini"]
    g_ch = models["challenger"]["discrimination"]["gini"]
    benchmark = {
        "scorecard_gini": g_sc,
        "challenger_gini": g_ch,
        "gini_gap": g_ch - g_sc,
        "gini_gap_pct": (g_ch - g_sc) / g_sc * 100 if g_sc else None,
        "verdict": _benchmark_verdict(g_sc, g_ch),
    }

    # ---- stability: score PSI across OOT vintages, feature CSI train-vs-test ---
    score_psi = stab.psi_by_vintage(scored, "credit_score", bins=int(v.psi_bins))
    split = make_split(cfg=cfg)
    feats = numeric_columns() + categorical_columns()
    feat_csi = stab.csi_by_feature(split.train, split.test, feats, bins=int(v.psi_bins))
    stability = {
        "score_psi_by_vintage": score_psi.to_dict(orient="records"),
        "max_score_psi": float(score_psi["psi"].max()),
        "feature_csi": feat_csi.to_dict(orient="records"),
        "max_feature_csi": float(feat_csi["csi"].max()),
    }

    report = {
        "stamp": run_stamp(),
        "n_oot": int(len(scored)),
        "oot_default_rate": float(y.mean()),
        "models": models,
        "benchmark": benchmark,
        "stability": stability,
        "thresholds": {
            "gini_min": float(v.gini_min), "ks_min": float(v.ks_min),
            "hl_pvalue_min": float(v.hl_pvalue_min),
            "calibration_max_band_error": float(v.calibration_max_band_error),
            "psi_watch": float(v.psi_watch), "psi_unstable": float(v.psi_unstable),
        },
    }

    out_json = cfg.path("artifacts") / "validation_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    _write_html(report, cfg.path("reports") / "validation_report.html")
    return report


def _benchmark_verdict(g_sc: float, g_ch: float) -> str:
    gap = g_ch - g_sc
    if gap <= 0.01:
        return ("Negligible accuracy gap — the interpretable scorecard is fully "
                "competitive. Deploy the scorecard; retain the challenger as an "
                "ongoing benchmark.")
    if gap <= 0.03:
        return ("Modest challenger edge. The interpretability/governance benefits "
                "of the scorecard likely outweigh the marginal Gini gain.")
    return ("Material challenger edge. Worth the model-risk cost of a less "
            "transparent model only with strong explainability controls (SHAP, "
            "reason codes) in place.")


def _badge(ok: bool) -> str:
    return ("<span class='ok'>PASS</span>" if ok
            else "<span class='bad'>REVIEW</span>")


def _write_html(r: dict, path: Path) -> None:
    t = r["thresholds"]
    rows = []
    for name in MODELS:
        d = r["models"][name]["discrimination"]
        c = r["models"][name]["calibration"]
        rows.append(f"""
        <tr><td>{name}</td>
            <td>{d['gini']:.4f} {_badge(d['gini'] >= t['gini_min'])}</td>
            <td>{d['ks']:.4f} {_badge(d['ks'] >= t['ks_min'])}</td>
            <td>{c['hosmer_lemeshow']['p_value']:.4f}</td>
            <td>{c['max_band_error']:.4f} {_badge(c['max_band_error'] <= t['calibration_max_band_error'])}</td>
            <td>{c['ece']:.4f}</td></tr>""")
    b = r["benchmark"]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CreditForge — Validation Report</title>
<style>
  body{{background:#0c0f14;color:#dfe6f0;font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:40px;}}
  h1{{font-weight:600;letter-spacing:.5px}} h2{{color:#7fb3ff;margin-top:32px;font-size:16px}}
  .mono,td,th{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
  table{{border-collapse:collapse;width:100%;margin-top:8px}}
  th,td{{padding:8px 12px;border-bottom:1px solid #1d2530;text-align:left}}
  th{{color:#8aa0b8;font-weight:500;text-transform:uppercase;font-size:11px;letter-spacing:.6px}}
  .ok{{color:#36d399;font-weight:600}} .bad{{color:#fb7185;font-weight:600}}
  .card{{background:#11161f;border:1px solid #1d2530;border-radius:10px;padding:18px 22px;margin-top:14px}}
  .muted{{color:#8aa0b8}} .accent{{color:#7fb3ff}}
</style></head><body>
<h1>CreditForge · Model Validation Report</h1>
<p class="muted mono">data={r['stamp']['data_version']} · code={r['stamp']['code_version']} · seed={r['stamp']['seed']}
   · OOT loans={r['n_oot']:,} · OOT default rate={r['oot_default_rate']:.2%}</p>

<h2>Discrimination &amp; Calibration (out-of-time)</h2>
<table><tr><th>Model</th><th>Gini</th><th>KS</th><th>HL p-value</th>
<th>Max band error</th><th>ECE</th></tr>{''.join(rows)}</table>

<h2>Scorecard vs Challenger Benchmark</h2>
<div class="card">
  <div class="mono">scorecard Gini = <span class="accent">{b['scorecard_gini']:.4f}</span>
   · challenger Gini = <span class="accent">{b['challenger_gini']:.4f}</span>
   · gap = {b['gini_gap']:+.4f} ({b['gini_gap_pct']:+.1f}%)</div>
  <p>{b['verdict']}</p>
</div>

<h2>Stability</h2>
<div class="card mono">
  max score PSI across vintages = {r['stability']['max_score_psi']:.4f}
   {_badge(r['stability']['max_score_psi'] <= t['psi_unstable'])}<br>
  max feature CSI (train→test) = {r['stability']['max_feature_csi']:.4f}
   {_badge(r['stability']['max_feature_csi'] <= t['psi_unstable'])}
</div>
<p class="muted">Thresholds: Gini ≥ {t['gini_min']}, KS ≥ {t['ks_min']},
  max band error ≤ {t['calibration_max_band_error']}, PSI watch {t['psi_watch']} / unstable {t['psi_unstable']}.</p>
</body></html>"""
    path.write_text(html)


if __name__ == "__main__":
    r = build_report()
    b = r["benchmark"]
    print(f"[validation] scorecard Gini={b['scorecard_gini']:.4f}  "
          f"challenger Gini={b['challenger_gini']:.4f}  gap={b['gini_gap']:+.4f}")
    print(f"[validation] max score PSI={r['stability']['max_score_psi']:.4f}  "
          f"max feature CSI={r['stability']['max_feature_csi']:.4f}")
    print(f"[validation] report -> artifacts/validation_report.json + reports/validation_report.html")
