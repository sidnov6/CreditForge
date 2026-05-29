"use client";
import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, ReferenceLine, Tooltip,
} from "recharts";
import { api, pct, money } from "@/lib/api";
import { PageHead, Stat, Band, Pill } from "@/components/ui";

const PRESETS = {
  "Prime borrower": { fico: 790, dti: 22, ltv: 60, orig_interest_rate: 3.1, orig_upb: 250000, orig_loan_term: 360, occupancy_status: "P", loan_purpose: "P", property_type: "SF", first_time_homebuyer: "N", vintage: "2021-06" },
  "Marginal borrower": { fico: 690, dti: 38, ltv: 82, orig_interest_rate: 4.3, orig_upb: 300000, orig_loan_term: 360, occupancy_status: "P", loan_purpose: "N", property_type: "CO", first_time_homebuyer: "Y", vintage: "2021-06" },
  "High-risk borrower": { fico: 645, dti: 47, ltv: 94, orig_interest_rate: 5.6, orig_upb: 360000, orig_loan_term: 360, occupancy_status: "I", loan_purpose: "C", property_type: "MH", first_time_homebuyer: "Y", vintage: "2021-06" },
};

const FIELDS = [
  ["fico", "FICO", "number"], ["dti", "DTI %", "number"], ["ltv", "LTV %", "number"],
  ["orig_interest_rate", "Note rate %", "number"], ["orig_upb", "Loan amount $", "number"],
  ["orig_loan_term", "Term (months)", "number"],
  ["occupancy_status", "Occupancy", ["P", "I", "S"]],
  ["loan_purpose", "Purpose", ["P", "C", "N"]],
  ["property_type", "Property", ["SF", "CO", "PU", "MH"]],
  ["first_time_homebuyer", "First-time buyer", ["N", "Y"]],
];

export default function ScorePage() {
  const [form, setForm] = useState(PRESETS["High-risk borrower"]);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      const payload = { ...form };
      ["fico", "ltv", "orig_upb", "orig_loan_term"].forEach((k) => (payload[k] = parseInt(payload[k])));
      ["dti", "orig_interest_rate"].forEach((k) => (payload[k] = parseFloat(payload[k])));
      setRes(await api.score(payload));
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const waterfall = res
    ? res.explanation.slice(0, 8).map((e) => ({ name: e.feature, shap: e.shap }))
    : [];

  return (
    <>
      <PageHead title="Score & Decision"
        subtitle="What happens to one borrower — PD, score, Expected Loss, and the reason codes behind the decision." />

      <div className="grid cols-2" style={{ alignItems: "start" }}>
        <div className="card">
          <h3>Applicant</h3>
          <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
            {Object.keys(PRESETS).map((p) => (
              <button key={p} className="pill neutral" style={{ cursor: "pointer" }}
                onClick={() => { setForm(PRESETS[p]); setRes(null); }}>{p}</button>
            ))}
          </div>
          <div className="form-grid">
            {FIELDS.map(([k, label, type]) => (
              <label key={k} className="field">
                {label}
                {Array.isArray(type) ? (
                  <select value={form[k]} onChange={(e) => set(k, e.target.value)}>
                    {type.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" value={form[k]} onChange={(e) => set(k, e.target.value)} />
                )}
              </label>
            ))}
          </div>
          <div style={{ marginTop: 16 }}>
            <button className="primary" onClick={submit} disabled={busy}>
              {busy ? "Scoring…" : "Score applicant"}
            </button>
          </div>
          {err && <p className="bad small mono" style={{ marginTop: 12 }}>{err}</p>}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {!res ? (
            <div className="card"><p className="muted">Submit an applicant to see the decision, the calibrated PD, the credit score and band, the Expected Loss, and the SHAP-driven adverse-action reasons.</p></div>
          ) : (
            <>
              <div className="card" style={{ borderColor: res.decision === "approve" ? "var(--green)" : "var(--red)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div className="label muted small" style={{ textTransform: "uppercase", letterSpacing: ".6px" }}>Decision</div>
                    <div style={{ fontSize: 28, fontWeight: 700, textTransform: "capitalize" }}
                         className={res.decision === "approve" ? "ok" : "bad"}>{res.decision}</div>
                  </div>
                  <Band grade={res.risk_band} />
                </div>
                <div className="small muted mono" style={{ marginTop: 8 }}>
                  threshold PD ≤ {pct(res.threshold)} · champion: calibrated challenger
                </div>
              </div>

              <div className="grid cols-3">
                <Stat label="Probability of Default" value={pct(res.pd)} tone={res.pd > res.threshold ? "bad" : "ok"} />
                <Stat label="Credit Score" value={res.credit_score} />
                <Stat label="Expected Loss" value={money(res.expected_loss)} sub={`LGD ${pct(res.lgd)} · EAD ${money(res.ead)}`} />
              </div>

              <div className="card">
                <h3>Reason codes — local SHAP attribution (red raises PD)</h3>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={waterfall} layout="vertical" margin={{ left: 30, right: 16 }}>
                    <XAxis type="number" stroke="#5c6b7e" />
                    <YAxis type="category" dataKey="name" width={120} stroke="#5c6b7e" />
                    <Tooltip formatter={(v) => v.toFixed(4)} />
                    <ReferenceLine x={0} stroke="#2b3645" />
                    <Bar dataKey="shap" radius={3}>
                      {waterfall.map((d, i) => (
                        <Cell key={i} fill={d.shap > 0 ? "#fb6f84" : "#36d399"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                {res.decision === "decline" && (
                  <ol style={{ margin: "8px 0 0", paddingLeft: 18 }} className="small">
                    {res.reason_codes.map((c) => <li key={c.rank} style={{ marginBottom: 3 }}>{c.reason}</li>)}
                  </ol>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
