"use client";
import {
  LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, ResponsiveContainer,
  Tooltip, ReferenceLine, Legend, CartesianGrid,
} from "recharts";
import { api, pct } from "@/lib/api";
import { PageHead, Stat, Pill, useFetch, Loading, ApiError } from "@/components/ui";

export default function ValidationPage() {
  const { data, error, loading } = useFetch(() => api.validation(), []);
  if (loading) return (<><Head /><Loading what="validation report" /></>);
  if (error) return (<><Head /><ApiError error={error} /></>);

  const t = data.thresholds;
  const sc = data.models.scorecard, ch = data.models.challenger;
  const b = data.benchmark;

  // reliability curve (challenger) — predicted vs observed, with diagonal
  const rel = ch.calibration.reliability.map((r) => ({ predicted: r.predicted, observed: r.observed }));
  const maxRel = Math.max(...rel.flatMap((r) => [r.predicted, r.observed]), 0.05);
  // gains capture curve (challenger)
  const gains = ch.discrimination.gains.map((g) => ({
    band: g.band, capture: g.cum_capture_rate, lift: g.lift, defaultRate: g.default_rate,
  }));

  const Metric = ({ label, sc, ch, fmt = (x) => x.toFixed(4), pass }) => (
    <tr>
      <td>{label}</td>
      <td className="num">{fmt(sc)}</td>
      <td className="num">{fmt(ch)}</td>
      <td>{pass && <Pill status={pass(ch)}>{pass(ch) ? "pass" : "review"}</Pill>}</td>
    </tr>
  );

  return (
    <>
      <Head />
      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Challenger Gini (OOT)" value={ch.discrimination.gini.toFixed(4)}
          tone={ch.discrimination.gini >= t.gini_min ? "ok" : "bad"} sub={`min ${t.gini_min}`} />
        <Stat label="KS statistic" value={ch.discrimination.ks.toFixed(4)}
          tone={ch.discrimination.ks >= t.ks_min ? "ok" : "bad"} sub={`min ${t.ks_min}`} />
        <Stat label="Calibration ECE" value={ch.calibration.ece.toFixed(4)} sub="lower is better" />
        <Stat label="OOT loans" value={data.n_oot.toLocaleString()}
          sub={`default rate ${pct(data.oot_default_rate)}`} />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Scorecard vs Challenger benchmark</h3>
        <div className="grid cols-3" style={{ gap: 12, marginBottom: 6 }}>
          <Stat label="Scorecard Gini" value={b.scorecard_gini.toFixed(4)} />
          <Stat label="Challenger Gini" value={b.challenger_gini.toFixed(4)} />
          <Stat label="Gini gap" value={`${b.gini_gap >= 0 ? "+" : ""}${b.gini_gap.toFixed(4)}`}
            sub={`${b.gini_gap_pct >= 0 ? "+" : ""}${b.gini_gap_pct.toFixed(1)}%`} />
        </div>
        <p className="small muted" style={{ marginTop: 4 }}>{b.verdict}</p>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Discrimination & calibration (out-of-time)</h3>
        <table>
          <thead><tr><th>Metric</th><th className="num">Scorecard</th><th className="num">Challenger</th><th>Gate</th></tr></thead>
          <tbody>
            <Metric label="Gini" sc={sc.discrimination.gini} ch={ch.discrimination.gini}
              pass={(x) => x >= t.gini_min} />
            <Metric label="AUC" sc={sc.discrimination.auc} ch={ch.discrimination.auc} />
            <Metric label="KS" sc={sc.discrimination.ks} ch={ch.discrimination.ks}
              pass={(x) => x >= t.ks_min} />
            <Metric label="Calibration max band error" sc={sc.calibration.max_band_error}
              ch={ch.calibration.max_band_error} pass={(x) => x <= t.calibration_max_band_error} />
            <Metric label="ECE" sc={sc.calibration.ece} ch={ch.calibration.ece} />
            <Metric label="Hosmer–Lemeshow p (informational)" sc={sc.calibration.hosmer_lemeshow.p_value}
              ch={ch.calibration.hosmer_lemeshow.p_value} />
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 8 }}>
          Calibration is gated on the economically meaningful max band error / ECE; with ~{(data.n_oot/1000).toFixed(0)}k
          loans the Hosmer–Lemeshow test over-rejects on trivially small miscalibration, so it is shown as informational.
        </p>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Reliability curve — challenger (predicted vs observed)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ left: 8, right: 12, top: 8 }}>
              <CartesianGrid stroke="#1d2530" />
              <XAxis type="number" dataKey="predicted" name="predicted" stroke="#5c6b7e"
                domain={[0, maxRel]} tickFormatter={(v) => pct(v, 0)} />
              <YAxis type="number" dataKey="observed" name="observed" stroke="#5c6b7e"
                domain={[0, maxRel]} tickFormatter={(v) => pct(v, 0)} />
              <Tooltip formatter={(v) => pct(v)} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: maxRel, y: maxRel }]} stroke="#2b3645" strokeDasharray="4 4" />
              <Scatter data={rel} fill="#5b9cff" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Cumulative gains — challenger</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={gains} margin={{ left: 8, right: 12, top: 8 }}>
              <CartesianGrid stroke="#1d2530" />
              <XAxis dataKey="band" stroke="#5c6b7e" label={{ value: "risk decile", position: "insideBottom", offset: -2, fill: "#5c6b7e", fontSize: 11 }} />
              <YAxis stroke="#5c6b7e" tickFormatter={(v) => pct(v, 0)} />
              <Tooltip formatter={(v) => pct(v)} />
              <Line type="monotone" dataKey="capture" stroke="#36d399" strokeWidth={2} dot={false}
                name="cumulative defaults captured" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

const Head = () => (
  <PageHead title="Model Validation"
    subtitle="The bank-credible centerpiece — Gini / KS, calibration, gains, and the scorecard-vs-challenger benchmark, all out-of-time." />
);
