"use client";
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, ReferenceLine,
} from "recharts";
import { api, pct } from "@/lib/api";
import { PageHead, Stat, Pill, Band, useFetch, Loading, ApiError } from "@/components/ui";

export default function GovernancePage() {
  const { data, error, loading } = useFetch(() => api.governance(), []);
  if (loading) return (<><Head /><Loading what="governance report" /></>);
  if (error) return (<><Head /><ApiError error={error} /></>);

  const f = data.fairness;
  const shap = data.shap_global.slice(0, 8).map((s) => ({ feature: s.feature, imp: s.mean_abs_shap }));
  const groups = [...f.groups].sort((a, b) => b.disparate_impact - a.disparate_impact);
  const ex = data.adverse_action_example;

  return (
    <>
      <Head />
      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Champion model" value="Challenger" sub="LightGBM · isotonic-calibrated" />
        <Stat label="Approval rate" value={pct(data.decision.approval_rate, 1)}
          sub={`PD threshold ${pct(data.decision.threshold)}`} />
        <Stat label="Min disparate-impact ratio" value={f.di_ratio_min_observed.toFixed(3)}
          tone={f.passes_four_fifths ? "ok" : "bad"} sub={`4/5ths floor ${f.di_ratio_floor}`} />
        <div className="card stat">
          <div className="label">Fairness (4/5ths rule)</div>
          <div style={{ marginTop: 10 }}>
            <Pill status={f.passes_four_fifths ? "PASS" : "REVIEW"}>
              {f.passes_four_fifths ? "pass" : "review"}
            </Pill>
          </div>
          <div className="sub">max EO diff {f.max_equal_opportunity_diff.toFixed(3)}</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Global drivers — mean |SHAP|</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={shap} layout="vertical" margin={{ left: 30, right: 12 }}>
              <XAxis type="number" stroke="#5c6b7e" />
              <YAxis type="category" dataKey="feature" width={120} stroke="#5c6b7e" />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="imp" fill="#5b9cff" radius={3} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Disparate impact by group (ratio vs {f.privileged_group})</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={groups} margin={{ left: 4, right: 12 }}>
              <XAxis dataKey="group" stroke="#5c6b7e" />
              <YAxis stroke="#5c6b7e" domain={[0, 1.2]} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <ReferenceLine y={f.di_ratio_floor} stroke="#fb6f84" strokeDasharray="4 4"
                label={{ value: "4/5ths floor", fill: "#fb6f84", fontSize: 10, position: "insideTopRight" }} />
              <Bar dataKey="disparate_impact" radius={3}>
                {groups.map((g, i) => (
                  <Cell key={i} fill={g.disparate_impact >= f.di_ratio_floor ? "#36d399" : "#fb6f84"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Fairness group metrics</h3>
        <table>
          <thead><tr>
            <th>Group</th><th className="num">N</th><th className="num">Approval</th>
            <th className="num">Default rate</th><th className="num">Disparate impact</th>
            <th className="num">Equal-opp. diff</th>
          </tr></thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.group}>
                <td>{g.group}</td>
                <td className="num">{g.n.toLocaleString()}</td>
                <td className="num">{pct(g.approval_rate, 1)}</td>
                <td className="num">{pct(g.default_rate, 2)}</td>
                <td className="num"><span className={g.disparate_impact >= f.di_ratio_floor ? "ok" : "bad"}>{g.disparate_impact.toFixed(3)}</span></td>
                <td className="num">{g.equal_opportunity_diff >= 0 ? "+" : ""}{g.equal_opportunity_diff.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 10 }}>
          The model excludes race, yet neutral features (credit score, DTI) proxy historical disparity, producing
          disparate impact. Mitigation: feature review, fairness-constrained training, monitoring — not group-specific
          thresholds (illegal in lending). Surfacing this is an EU AI Act high-risk expectation.
        </p>
      </div>

      {ex && (
        <div className="card">
          <h3>Worked adverse-action example</h3>
          <div style={{ display: "flex", gap: 24, alignItems: "center", marginBottom: 12 }}>
            <span className="mono small muted">loan {ex.loan_id}</span>
            <span>PD <b className="mono">{pct(ex.pd)}</b></span>
            <span>score <b className="mono">{ex.credit_score.toFixed(0)}</b></span>
            <Band grade={ex.band} />
            <Pill status={false}>decline</Pill>
          </div>
          <pre className="mono small" style={{ whiteSpace: "pre-wrap", background: "var(--panel-2)", padding: 14, borderRadius: 8, border: "1px solid var(--border)" }}>{ex.letter}</pre>
        </div>
      )}
    </>
  );
}

const Head = () => (
  <PageHead title="Governance"
    subtitle="Explainability, fairness, and accountability — global SHAP drivers, disparate-impact testing, and the model card in one place." />
);
