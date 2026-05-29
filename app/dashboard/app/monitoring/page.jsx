"use client";
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, ReferenceLine,
} from "recharts";
import { api } from "@/lib/api";
import { PageHead, Stat, Pill, useFetch, Loading, ApiError } from "@/components/ui";

const psiColor = (s) => (s === "stable" ? "#36d399" : s === "watch" ? "#fbbf24" : "#fb6f84");

export default function MonitoringPage() {
  const v = useFetch(() => api.validation(), []);
  const m = useFetch(() => api.monitoring(), []);
  if (v.loading || m.loading) return (<><Head /><Loading what="stability metrics" /></>);
  if (v.error) return (<><Head /><ApiError error={v.error} /></>);
  if (m.error) return (<><Head /><ApiError error={m.error} /></>);

  const t = v.data.thresholds;
  const psiByVintage = v.data.stability.score_psi_by_vintage.map((r) => ({
    vintage: r.vintage, psi: r.psi, status: r.status,
  }));
  const csi = v.data.stability.feature_csi.map((r) => ({ feature: r.feature, csi: r.csi, status: r.status }));
  const mon = m.data;

  return (
    <>
      <Head />
      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Max score PSI (vintages)" value={v.data.stability.max_score_psi.toFixed(4)}
          tone={v.data.stability.max_score_psi <= t.psi_watch ? "ok" : v.data.stability.max_score_psi <= t.psi_unstable ? "warn" : "bad"}
          sub={`watch ${t.psi_watch} · unstable ${t.psi_unstable}`} />
        <Stat label="Max feature CSI" value={v.data.stability.max_feature_csi.toFixed(4)}
          tone={v.data.stability.max_feature_csi <= t.psi_watch ? "ok" : "warn"} />
        <Stat label="Live batch score PSI" value={mon.score_psi?.toFixed(4) ?? "—"}
          tone={mon.score_status === "stable" ? "ok" : mon.score_status === "watch" ? "warn" : "bad"}
          sub={`vs training baseline`} />
        <div className="card stat">
          <div className="label">Monitoring status</div>
          <div style={{ marginTop: 10 }}>
            <Pill status={mon.healthy}>{mon.healthy ? "healthy" : `${mon.alerts.length} alert(s)`}</Pill>
          </div>
          <div className="sub">{mon.n_batch.toLocaleString()} loans monitored</div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Score PSI across vintages (vs earliest cohort)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={psiByVintage} margin={{ left: 4, right: 8 }}>
            <XAxis dataKey="vintage" stroke="#5c6b7e" interval={2} angle={-30} textAnchor="end" height={50} />
            <YAxis stroke="#5c6b7e" />
            <Tooltip formatter={(val) => val.toFixed(4)} />
            <ReferenceLine y={t.psi_watch} stroke="#fbbf24" strokeDasharray="4 4" label={{ value: "watch", fill: "#fbbf24", fontSize: 10, position: "right" }} />
            <ReferenceLine y={t.psi_unstable} stroke="#fb6f84" strokeDasharray="4 4" label={{ value: "unstable", fill: "#fb6f84", fontSize: 10, position: "right" }} />
            <Bar dataKey="psi" radius={3}>
              {psiByVintage.map((d, i) => <Cell key={i} fill={psiColor(d.status)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Characteristic Stability (CSI) by feature — train → test</h3>
          <table>
            <thead><tr><th>Feature</th><th className="num">CSI</th><th>Status</th></tr></thead>
            <tbody>
              {csi.slice(0, 10).map((r) => (
                <tr key={r.feature}>
                  <td className="mono small">{r.feature}</td>
                  <td className="num">{r.csi.toFixed(4)}</td>
                  <td><Pill status={r.status}>{r.status}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h3>Drift alerts (live batch vs baseline)</h3>
          {mon.alerts.length === 0 ? (
            <p className="ok small">No drift alerts — all signals within stable thresholds. The model is healthy on the monitored batch.</p>
          ) : (
            <table>
              <thead><tr><th>Signal</th><th className="num">PSI</th><th>Status</th></tr></thead>
              <tbody>
                {mon.alerts.map((a, i) => (
                  <tr key={i}><td className="mono small">{a.signal}</td>
                    <td className="num">{a.psi}</td><td><Pill status={a.status}>{a.status}</Pill></td></tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="small muted" style={{ marginTop: 12 }}>
            PSI/CSI re-computed against the training-time baseline. Thresholds: &lt; {t.psi_watch} stable,
            {t.psi_watch}–{t.psi_unstable} watch, &gt; {t.psi_unstable} unstable. Runs on a schedule (GitHub Actions cron).
          </p>
        </div>
      </div>
    </>
  );
}

const Head = () => (
  <PageHead title="Stability & Monitoring"
    subtitle="Is the model still healthy — PSI/CSI over time with threshold bands and drift alerts against the training baseline." />
);
