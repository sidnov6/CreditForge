"use client";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Cell,
  ResponsiveContainer,
} from "recharts";

const fmt = (v, kind) => {
  if (v == null) return "—";
  if (kind === "pct") return `${(v * 100).toFixed(2)}%`;
  if (kind === "money") {
    const a = Math.abs(v);
    if (a >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
    return `$${v.toFixed(0)}`;
  }
  return typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4)) : v;
};

const AXIS = "#5c6b7e";
const GREEN = "#36d399", RED = "#fb6f84", ACCENT = "#5b9cff";

export default function AgentChart({ spec }) {
  if (!spec) return null;
  const vf = spec.valueFormat || "num";

  const Frame = ({ children, height = 240 }) => (
    <div className="card" style={{ marginTop: 10, background: "var(--panel-2)" }}>
      <h3 style={{ marginBottom: 10 }}>{spec.title}</h3>
      <ResponsiveContainer width="100%" height={height}>{children}</ResponsiveContainer>
    </div>
  );

  if (spec.type === "bar") {
    const s = spec.series[0];
    return (
      <Frame>
        <BarChart data={spec.data} margin={{ left: 10, right: 12, top: 6 }}>
          <CartesianGrid stroke="#1d2530" vertical={false} />
          <XAxis dataKey={spec.x} stroke={AXIS} tick={{ fontSize: 11 }} />
          <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v, vf)} width={64} />
          <Tooltip formatter={(v) => fmt(v, vf)} />
          {spec.reference && (
            <ReferenceLine y={spec.reference.value} stroke={RED} strokeDasharray="4 4"
              label={{ value: spec.reference.label, fill: RED, fontSize: 10, position: "insideTopRight" }} />
          )}
          <Bar dataKey={s.key} radius={4}>
            {spec.data.map((_, i) => (
              <Cell key={i} fill={spec.colors?.[i] || s.color || ACCENT} />
            ))}
          </Bar>
        </BarChart>
      </Frame>
    );
  }

  if (spec.type === "hbar") {
    const h = Math.max(160, spec.data.length * 30 + 40);
    return (
      <Frame height={h}>
        <BarChart data={spec.data} layout="vertical" margin={{ left: 30, right: 16 }}>
          <XAxis type="number" stroke={AXIS} tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v, vf)} />
          <YAxis type="category" dataKey="name" width={130} stroke={AXIS} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v) => fmt(v, vf)} />
          {spec.diverging && <ReferenceLine x={0} stroke="#2b3645" />}
          <Bar dataKey="value" radius={3}>
            {spec.data.map((d, i) => (
              <Cell key={i} fill={spec.diverging ? (d.value > 0 ? RED : GREEN) : ACCENT} />
            ))}
          </Bar>
        </BarChart>
      </Frame>
    );
  }

  if (spec.type === "line") {
    return (
      <Frame>
        <LineChart data={spec.data} margin={{ left: 10, right: 12, top: 6 }}>
          <CartesianGrid stroke="#1d2530" />
          <XAxis dataKey={spec.x} stroke={AXIS} tick={{ fontSize: 11 }} />
          <YAxis stroke={AXIS} tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v, vf)} width={64} />
          <Tooltip formatter={(v) => fmt(v, vf)} />
          {spec.reference && (
            <ReferenceLine y={spec.reference.value} stroke={RED} strokeDasharray="4 4"
              label={{ value: spec.reference.label, fill: RED, fontSize: 10, position: "right" }} />
          )}
          {spec.series.map((s) => (
            <Line key={s.key} type="monotone" dataKey={s.key} stroke={s.color || ACCENT}
              strokeWidth={2} dot={false} name={s.label} />
          ))}
        </LineChart>
      </Frame>
    );
  }

  if (spec.type === "scatter") {
    const d = spec.domain || [0, "auto"];
    return (
      <Frame>
        <ScatterChart margin={{ left: 10, right: 12, top: 6 }}>
          <CartesianGrid stroke="#1d2530" />
          <XAxis type="number" dataKey="x" stroke={AXIS} domain={d}
            tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v, vf)} name={spec.xLabel} />
          <YAxis type="number" dataKey="y" stroke={AXIS} domain={d}
            tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v, vf)} name={spec.yLabel} />
          <Tooltip formatter={(v) => fmt(v, vf)} />
          {spec.diagonal && (
            <ReferenceLine segment={[{ x: d[0], y: d[0] }, { x: d[1], y: d[1] }]}
              stroke="#2b3645" strokeDasharray="4 4" />
          )}
          <Scatter data={spec.data} fill={ACCENT} />
        </ScatterChart>
      </Frame>
    );
  }
  return null;
}
