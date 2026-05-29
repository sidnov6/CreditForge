"use client";
import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV = [
  { href: "/", label: "Score & Decision" },
  { href: "/portfolio", label: "Portfolio Risk" },
  { href: "/validation", label: "Model Validation" },
  { href: "/monitoring", label: "Stability & Monitoring" },
  { href: "/governance", label: "Governance" },
  { href: "/copilot", label: "Risk Copilot", accent: true },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="name">CreditForge</span>
        <span className="code">BONITAS</span>
        <span className="tag">PD · LGD · EAD · Expected Loss</span>
      </div>
      <nav className="nav">
        {NAV.map((n) => {
          const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
          return (
            <Link key={n.href} href={n.href} className={active ? "active" : ""}
              style={n.accent && !active ? { color: "var(--accent)" } : undefined}>
              <span className="dot" /> {n.label}
              {n.accent && <span style={{ marginLeft: "auto", fontSize: 9, opacity: .8,
                border: "1px solid currentColor", borderRadius: 5, padding: "1px 5px",
                letterSpacing: ".5px" }}>AI</span>}
            </Link>
          );
        })}
      </nav>
      <div style={{ position: "absolute", bottom: 18, left: 24, right: 24 }}
           className="small muted mono">
        out-of-time validated<br />free-tier · $0 infra
      </div>
    </aside>
  );
}
