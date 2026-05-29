"""Silver layer — cleaning + the leakage-safe point-in-time default flag.

This is the keystone of the whole platform. Two disciplines are enforced here:

1. **Point-in-time, forward-looking target.** The observation date is loan
   origination. The 12-month default flag looks at performance rows strictly
   inside the forward window (origination, origination + 12 months]. A loan
   defaults if it hits 90+ DPD (>= 3 months delinquent) or an REO disposition
   in that window. Nothing after the window can influence the label, and no
   performance information enters the feature set at all.

2. **Data-quality gates.** Malformed/implausible origination rows are rejected
   before they can poison training (Part 10.6).

The realized loss + exposure-at-default are also carried through for LGD.
DuckDB does the heavy lifting straight off the Bronze parquet (no server).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from creditforge.config import Config, load_config

# Plausibility ranges for the Silver data-quality gate (reject outside these)
QUALITY_RANGES = {
    "fico": (300, 850),
    "dti": (0, 70),
    "ltv": (1, 125),
    "orig_interest_rate": (0, 25),
    "orig_upb": (1_000, 5_000_000),
    "orig_loan_term": (60, 480),
}


def build_silver(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    bronze = cfg.path("bronze")
    silver = cfg.path("silver")
    silver.mkdir(parents=True, exist_ok=True)

    horizon = int(cfg.target.observation_horizon_months)
    dpd_months = int(cfg.target.default_dpd_threshold) // 30  # 90 days -> 3 months

    orig_glob = str(bronze / "origination" / "*.parquet")
    perf_glob = str(bronze / "performance" / "*.parquet")

    con = duckdb.connect()
    # Month index helper: year*12 + month, computed from 'YYYY-MM' strings.
    con.execute("""
        CREATE MACRO midx(s) AS
            CAST(SUBSTR(s, 1, 4) AS INT) * 12 + CAST(SUBSTR(s, 6, 2) AS INT);
    """)

    # ---- point-in-time forward join: default events strictly within window ----
    con.execute(f"""
        CREATE TEMP TABLE perf_events AS
        SELECT
            p.loan_id,
            p.monthly_reporting_period,
            midx(p.monthly_reporting_period)              AS rep_idx,
            TRY_CAST(p.current_loan_delinquency_status AS INT) AS dpd_months,
            p.zero_balance_code,
            p.current_upb,
            p.net_loss
        FROM read_parquet('{perf_glob}') p;
    """)

    con.execute(f"""
        CREATE TEMP TABLE orig AS
        SELECT *, midx(vintage) AS vint_idx
        FROM read_parquet('{orig_glob}');
    """)

    # For each loan: did a default event occur in (obs, obs+horizon]? when? loss?
    con.execute(f"""
        CREATE TEMP TABLE dflt AS
        WITH win AS (
            SELECT e.loan_id, e.rep_idx, e.zero_balance_code,
                   e.current_upb, e.net_loss,
                   (e.dpd_months >= {dpd_months} OR e.zero_balance_code = '09') AS is_def_event
            FROM perf_events e
            JOIN orig o ON e.loan_id = o.loan_id
            WHERE e.rep_idx > o.vint_idx
              AND e.rep_idx <= o.vint_idx + {horizon}
        ),
        first_def AS (
            SELECT loan_id, MIN(rep_idx) AS def_idx
            FROM win WHERE is_def_event GROUP BY loan_id
        )
        SELECT
            f.loan_id,
            f.def_idx,
            -- exposure at default = current balance at the first default month
            (SELECT w.current_upb FROM win w
              WHERE w.loan_id = f.loan_id AND w.rep_idx = f.def_idx LIMIT 1) AS ead,
            -- realized loss from the REO disposition row, if any, within window
            (SELECT MAX(w.net_loss) FROM win w WHERE w.loan_id = f.loan_id) AS net_loss
        FROM first_def f;
    """)

    silver_df = con.execute("""
        SELECT
            o.* EXCLUDE (vint_idx),
            CASE WHEN d.loan_id IS NOT NULL THEN 1 ELSE 0 END AS default_12m,
            d.ead,
            d.net_loss
        FROM orig o
        LEFT JOIN dflt d ON o.loan_id = d.loan_id
        ORDER BY o.loan_id;
    """).df()
    con.close()

    # ---- data-quality gate: reject implausible rows before they poison training
    before = len(silver_df)
    mask = pd.Series(True, index=silver_df.index)
    for col, (lo, hi) in QUALITY_RANGES.items():
        mask &= silver_df[col].between(lo, hi)
    rejected = int((~mask).sum())
    silver_df = silver_df[mask].reset_index(drop=True)

    out = silver / "loans.parquet"
    silver_df.to_parquet(out, index=False)

    rate = silver_df["default_12m"].mean()
    print(f"[silver] {before:,} loans in, {rejected:,} rejected by quality gate, "
          f"{len(silver_df):,} out | 12m default rate {rate:.2%}")
    return out


if __name__ == "__main__":
    build_silver()
