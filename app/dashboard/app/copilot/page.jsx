"use client";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { PageHead, Pill } from "@/components/ui";
import AgentChart from "@/components/AgentChart";

const SUGGESTIONS = [
  "Summarize the portfolio's risk and Expected Loss",
  "Which FICO bands default the most?",
  "Is the model valid? Show Gini and calibration.",
  "Is the model fair across racial groups, and how would we mitigate it?",
  "Score a borrower: FICO 650, DTI 46, LTV 93, investment property, cash-out",
  "Why is the 2021 vintage riskier, and is the model still stable?",
];

// minimal markdown: **bold**, bullets, paragraphs
function RichText({ text }) {
  const blocks = (text || "").split("\n");
  return (
    <div>
      {blocks.map((ln, i) => {
        const bullet = /^\s*[-*]\s+/.test(ln);
        const html = ln.replace(/^\s*[-*]\s+/, "")
          .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
          .replace(/`(.+?)`/g, '<code>$1</code>');
        if (!ln.trim()) return <div key={i} style={{ height: 6 }} />;
        return bullet
          ? <div key={i} style={{ display: "flex", gap: 8 }}>
              <span className="accent">•</span>
              <span dangerouslySetInnerHTML={{ __html: html }} />
            </div>
          : <p key={i} style={{ margin: "4px 0" }} dangerouslySetInnerHTML={{ __html: html }} />;
      })}
    </div>
  );
}

function AgentChips({ team, active }) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {team.map((a) => {
        const on = active?.some((x) => x.specialist === a.id);
        return (
          <span key={a.id} className={`pill ${on ? "ok" : "neutral"}`}
            title={a.tools.join(", ")}>{a.title}{on ? " ✓" : ""}</span>
        );
      })}
    </div>
  );
}

function Trace({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace?.length) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <button className="pill neutral" style={{ cursor: "pointer" }} onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} agent trace · {trace.length} tool call{trace.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div className="mono small" style={{ marginTop: 8, borderLeft: "2px solid var(--border-bright)", paddingLeft: 12 }}>
          {trace.map((t, i) => (
            <div key={i} style={{ marginBottom: 4 }}>
              <span className="accent">{t.agent}</span>
              <span className="muted"> → </span>{t.tool}
              <span className="muted">({Object.entries(t.args || {}).map(([k, v]) => `${k}=${v}`).join(", ")})</span>
              {t.chart && <span className="ok"> ▣ chart</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function CopilotPage() {
  const [team, setTeam] = useState([]);
  const [ready, setReady] = useState(null);
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    api.agentTeam().then((t) => { setTeam(t.agents); setReady(t.llm_ready); setModel(t.model); })
      .catch(() => setReady(false));
  }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  const ask = async (q) => {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const r = await api.agentChat(question);
      setMessages((m) => [...m, { role: "assistant", text: r.answer,
        agents: r.agents, charts: r.charts, trace: r.trace }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", error: true, text: e.message }]);
    } finally { setBusy(false); }
  };

  return (
    <>
      <PageHead title="Risk Copilot"
        subtitle="Ask the specialist agent team — they call the real PD/LGD/EL tools and answer with live charts." />

      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <h3 style={{ margin: 0 }}>Agent team</h3>
          <span className="small mono muted">
            {ready === null ? "…" : ready
              ? <>orchestrator · {model} <Pill status={true}>online</Pill></>
              : <Pill status={false}>LLM key not set</Pill>}
          </span>
        </div>
        <AgentChips team={team} active={null} />
        {ready === false && (
          <p className="small muted" style={{ marginTop: 10 }}>
            Set a free <span className="mono">GROQ_API_KEY</span> (console.groq.com) on the
            server to enable the agents. The charts and metrics on the other screens work without it.
          </p>
        )}
      </div>

      <div className="card" style={{ minHeight: 320 }}>
        {messages.length === 0 && (
          <div className="muted" style={{ marginBottom: 14 }}>
            <p style={{ marginTop: 0 }}>Try one of these — each answer is produced by the agents calling real tools, with charts:</p>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {SUGGESTIONS.map((s) => (
                <button key={s} className="pill neutral" style={{ cursor: "pointer" }}
                  onClick={() => ask(s)} disabled={busy}>{s}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 18 }}>
            {m.role === "user" ? (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <div className="card" style={{ background: "var(--accent-dim)", padding: "8px 14px", maxWidth: "80%" }}>
                  {m.text}
                </div>
              </div>
            ) : (
              <div className="card" style={{ background: "var(--panel-2)" }}>
                {m.agents?.length > 0 && (
                  <div style={{ marginBottom: 10 }}><AgentChips team={team} active={m.agents} /></div>
                )}
                <div className={m.error ? "bad" : ""}><RichText text={m.text} /></div>
                {m.charts?.map((c) => <AgentChart key={c.id} spec={c} />)}
                <Trace trace={m.trace} />
              </div>
            )}
          </div>
        ))}

        {busy && (
          <div className="card" style={{ background: "var(--panel-2)" }}>
            <span className="mono small accent">agents analyzing</span>
            <span className="mono small muted dots"> · routing → calling tools → synthesizing…</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
        <input style={{ flex: 1, fontFamily: "var(--sans)" }}
          placeholder="Ask the risk team… (e.g. what drives Expected Loss in the book?)"
          value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()} disabled={busy || ready === false} />
        <button className="primary" onClick={() => ask()} disabled={busy || ready === false}>
          {busy ? "…" : "Ask"}
        </button>
      </div>
    </>
  );
}
