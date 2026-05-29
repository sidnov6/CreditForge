"use client";
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell,
} from "recharts";
import { api, pct, money, moneyShort, num } from "@/lib/api";
import { PageHead, Stat, Band, useFetch, Loading, ApiError } from "@/components/ui";

const BAND_COLOR = { AAA: "#36d399", AA: "#36d399", A: "#5fcf9e", BBB: "#5b9cff", BB: "#fbbf24", B: "#fb6f84" };

export default function PortfolioPage() {
  const { data, error, loading } = useFetch(() => api.portfolio(300), []);
  if (loading) return (<><Head /><Loading what="portfolio" /></>);
  if (error) return (<><Head /><ApiError error={error} /></>);

  const bandData = Object.entries(data.band_distribution).map(([k, v]) => ({
    band: k, count: v, el: data.el_by_band[k] || 0,
  }));
  const hist = data.pd_histogram.map((h) => ({ pd: (h.x * 100).toFixed(1), count: h.count }));

  return (
    <>
      <Head />
      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <Stat label="Loans (out-of-time book)" value={num(data.n_loans)} />
        <Stat label="Portfolio Expected Loss" value={moneyShort(data.portfolio_el)}
          sub={`${data.el_rate_bps.toFixed(0)} bps of EAD`} tone="warn" />
        <Stat label="Total EAD" value={moneyShort(data.total_ead)} />
        <Stat label="Approval rate" value={pct(data.approval_rate, 1)}
          sub={`mean PD ${pct(data.mean_pd)}`} />
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Risk-band concentration</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={bandData} margin={{ left: 4, right: 8 }}>
              <XAxis dataKey="band" stroke="#5c6b7e" />
              <YAxis stroke="#5c6b7e" />
              <Tooltip formatter={(v) => num(v)} />
              <Bar dataKey="count" radius={4}>
                {bandData.map((d, i) => <Cell key={i} fill={BAND_COLOR[d.band] || "#5b9cff"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3>Expected Loss by band</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={bandData} margin={{ left: 14, right: 8 }}>
              <XAxis dataKey="band" stroke="#5c6b7e" />
              <YAxis stroke="#5c6b7e" tickFormatter={moneyShort} />
              <Tooltip formatter={(v) => money(v)} />
              <Bar dataKey="el" radius={4}>
                {bandData.map((d, i) => <Cell key={i} fill={BAND_COLOR[d.band] || "#5b9cff"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>PD distribution across the book</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={hist} margin={{ left: 4, right: 8 }}>
            <XAxis dataKey="pd" stroke="#5c6b7e" tickFormatter={(v) => `${v}%`}
              interval={3} />
            <YAxis stroke="#5c6b7e" />
            <Tooltip formatter={(v) => num(v)} labelFormatter={(l) => `PD ≈ ${l}%`} />
            <Bar dataKey="count" fill="#5b9cff" radius={2} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Highest Expected-Loss exposures</h3>
        <table>
          <thead><tr>
            <th>Loan</th><th>Vintage</th><th className="num">PD</th><th className="num">Score</th>
            <th>Band</th><th className="num">LGD</th><th className="num">EAD</th>
            <th className="num">Expected Loss</th><th>Decision</th>
          </tr></thead>
          <tbody>
            {data.sample.slice(0, 18).map((r) => (
              <tr key={r.loan_id}>
                <td className="mono">{r.loan_id}</td>
                <td className="mono">{r.vintage}</td>
                <td className="num">{pct(r.pd_challenger)}</td>
                <td className="num">{r.credit_score.toFixed(0)}</td>
                <td><Band grade={r.risk_band} /></td>
                <td className="num">{pct(r.lgd_hat)}</td>
                <td className="num">{moneyShort(r.ead)}</td>
                <td className="num">{money(r.expected_loss)}</td>
                <td className={r.decision === "approve" ? "ok small" : "bad small"}>{r.decision}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

const Head = () => (
  <PageHead title="Portfolio Risk"
    subtitle="What the book looks like — Expected Loss totals, risk-band concentration, and the PD distribution on the out-of-time portfolio." />
);
