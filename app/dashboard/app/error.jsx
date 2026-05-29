"use client";
// Route-level error boundary: a render/runtime error shows this card instead of
// silently unmounting the tree into a blank screen.
export default function Error({ error, reset }) {
  return (
    <div className="card" style={{ borderColor: "var(--red)", marginTop: 20 }}>
      <h3 className="bad">Something went wrong rendering this screen</h3>
      <p className="small muted mono" style={{ whiteSpace: "pre-wrap" }}>
        {error?.message || String(error)}
      </p>
      <p className="small muted">
        Most often this means the scoring API isn’t reachable. Make sure it’s running:
        <br />
        <span className="mono">uvicorn app.api.main:app --port 8001</span>
      </p>
      <button className="primary" onClick={() => reset()} style={{ marginTop: 10 }}>
        Retry
      </button>
    </div>
  );
}
