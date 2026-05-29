// Thin client for the CreditForge FastAPI scoring service.
// Same-origin in the container (base = "/api"); in split local dev set
// NEXT_PUBLIC_API_URL=http://localhost:8001 and it becomes that host + "/api".
export const API = `${process.env.NEXT_PUBLIC_API_URL || ""}/api`;

async function get(path) {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
  return r.json();
}

export const api = {
  health: () => get("/health"),
  validation: () => get("/validation"),
  governance: () => get("/governance"),
  monitoring: () => get("/monitoring"),
  shapGlobal: () => get("/shap/global"),
  portfolio: (n = 400) => get(`/portfolio?n=${n}`),
  agentTeam: () => get("/agent/team"),
  agentChat: async (question) => {
    const r = await fetch(`${API}/agent/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `agent -> HTTP ${r.status}`);
    return body;
  },
  score: async (applicant) => {
    const r = await fetch(`${API}/score`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(applicant),
    });
    if (!r.ok) throw new Error(`score -> HTTP ${r.status}`);
    return r.json();
  },
};

// formatters
export const pct = (x, d = 2) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
export const num = (x, d = 0) =>
  x == null ? "—" : x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
export const money = (x) =>
  x == null ? "—" : `$${Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
export const moneyShort = (x) => {
  if (x == null) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return `$${(x / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `$${(x / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `$${(x / 1e3).toFixed(1)}K`;
  return `$${x.toFixed(0)}`;
};

// PSI / generic status -> semantic class
export function statusClass(status) {
  return status === "stable" || status === true ? "ok"
    : status === "watch" ? "warn" : "bad";
}
