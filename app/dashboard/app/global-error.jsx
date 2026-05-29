"use client";
// Catches errors in the root layout itself (must render its own <html>/<body>).
export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body style={{ background: "#0a0d12", color: "#dfe6f0", fontFamily: "sans-serif", padding: 40 }}>
        <h2>Application error</h2>
        <p style={{ color: "#8090a3", fontFamily: "monospace" }}>{error?.message || String(error)}</p>
        <button onClick={() => reset()}
          style={{ background: "#5b9cff", color: "#06101f", border: "none", borderRadius: 8, padding: "10px 18px", fontWeight: 600, cursor: "pointer" }}>
          Reload
        </button>
      </body>
    </html>
  );
}
