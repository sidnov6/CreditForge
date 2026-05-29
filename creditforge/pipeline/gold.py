"""Gold layer — the model-ready feature matrix.

Projects Silver into exactly what the models consume: the configured numeric +
categorical features, a handful of **origination-only** engineered features (so
the leakage guarantee holds), the 12-month target, the vintage tag for the
out-of-time split, the protected attribute (carried for fairness, never fed to
a model), and the LGD fields (EAD, realized loss, derived LGD on defaults).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features — strictly from information known at origination."""
    df = df.copy()
    # LTV x DTI interaction: thin-equity AND stretched borrower compound risk
    df["ltv_dti"] = (df["ltv"] * df["dti"] / 100.0).round(2)
    # Rate spread vs the borrower's own vintage average (a vintage stat is known
    # at origination, so no look-ahead): isolates idiosyncratic pricing/risk.
    vint_rate = df.groupby("vintage")["orig_interest_rate"].transform("mean")
    df["rate_spread"] = (df["orig_interest_rate"] - vint_rate).round(3)
    # High-LTV flag (regulatory PMI boundary at 80)
    df["high_ltv"] = (df["ltv"] > 80).astype(int)
    return df


def build_gold(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    silver = cfg.path("silver") / "loans.parquet"
    gold = cfg.path("gold")
    gold.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(silver)
    df = _engineer(df)

    # Derived LGD on the defaulted population: bounded [0, 1]
    df["lgd"] = np.where(
        (df["default_12m"] == 1) & df["ead"].gt(0),
        (df["net_loss"] / df["ead"]).clip(0, 1),
        np.nan,
    )

    numeric = list(cfg.features.numeric) + ["ltv_dti", "rate_spread", "high_ltv"]
    categorical = list(cfg.features.categorical)
    protected = list(cfg.features.protected)
    keep = (["loan_id", "vintage", "default_12m"] + numeric + categorical
            + protected + ["ead", "net_loss", "lgd"])

    out_df = df[keep].copy()
    # Cast categoricals so downstream (LightGBM) can use native categorical dtype
    for c in categorical:
        out_df[c] = out_df[c].astype("category")

    out = gold / "feature_matrix.parquet"
    out_df.to_parquet(out, index=False)

    # Persist the feature manifest so models/serving agree on inputs
    manifest = {"numeric": numeric, "categorical": categorical,
                "protected": protected, "target": "default_12m"}
    import json
    (gold / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[gold] {len(out_df):,} rows | {len(numeric)} numeric + "
          f"{len(categorical)} categorical features | "
          f"{out_df['lgd'].notna().sum():,} loans with realized LGD")
    return out


if __name__ == "__main__":
    build_gold()
