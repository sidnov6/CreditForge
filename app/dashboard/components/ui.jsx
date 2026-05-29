"use client";
import { useEffect, useState } from "react";

export function Stat({ label, value, sub, tone }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className={`value ${tone || ""}`}>{value}</div>
      {sub != null && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Pill({ status, children }) {
  const cls = status === true || status === "stable" || status === "PASS" ? "ok"
    : status === "watch" ? "warn"
    : status === false || status === "unstable" || status === "REVIEW" ? "bad"
    : "neutral";
  return <span className={`pill ${cls}`}>{children}</span>;
}

export function Band({ grade }) {
  return <span className={`band ${grade}`}>{grade}</span>;
}

export function PageHead({ title, subtitle }) {
  return (
    <div className="page-head">
      <h1>{title}</h1>
      {subtitle && <p>{subtitle}</p>}
    </div>
  );
}

// data-fetching hook with loading / error states
export function useFetch(fn, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    setState({ data: null, error: null, loading: true });
    fn()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error: error.message, loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function Loading({ what = "data" }) {
  return <div className="loading">loading {what}…</div>;
}

export function ApiError({ error }) {
  return (
    <div className="card" style={{ borderColor: "var(--red)" }}>
      <h3 className="bad">Cannot reach scoring API</h3>
      <p className="small muted">{error}</p>
      <p className="small mono">
        Start it with:&nbsp; uvicorn app.api.main:app --port 8001
        <br />(after running: python -m creditforge.cli all)
      </p>
    </div>
  );
}
