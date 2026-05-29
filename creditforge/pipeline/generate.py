"""Synthetic GSE-schema data generator.

Reproduces the *shape* of the Freddie Mac Single-Family Loan-Level Dataset:
an **origination** table (one row per loan, attributes known at origination)
and a **monthly performance** panel (one row per loan-month, with delinquency
status, current balance, zero-balance disposition codes, and realized net loss).

Defaults are driven by a latent logit of the real risk features, so a model has
genuine signal to learn. Losses (for LGD) and the monthly delinquency march are
simulated consistently. Everything is seeded -> bit-for-bit reproducible.

Real Freddie/Fannie files drop into Silver unchanged: they carry the same
columns this generator emits (see `ORIG_SCHEMA` / `PERF_SCHEMA`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config, set_global_seed

# --- Schema (subset of the Freddie Mac data dictionary we actually model) ----
ORIG_SCHEMA = [
    "loan_id", "vintage", "orig_date",
    "fico", "dti", "ltv", "orig_interest_rate", "orig_upb", "orig_loan_term",
    "occupancy_status", "loan_purpose", "property_type", "first_time_homebuyer",
    "borrower_race",   # protected attribute for fairness — NEVER a model input
]
PERF_SCHEMA = [
    "loan_id", "monthly_reporting_period", "loan_age",
    "current_upb", "current_loan_delinquency_status",
    "zero_balance_code", "net_loss",
]

# Freddie zero-balance codes we use: 01 prepaid (no loss), 09 REO disposition (loss)
ZB_PREPAID = "01"
ZB_REO = "09"

_OCCUPANCY = np.array(["P", "I", "S"])       # Primary, Investment, Second home
_PURPOSE = np.array(["P", "C", "N"])         # Purchase, Cash-out refi, No-cash refi
_PROPERTY = np.array(["SF", "CO", "PU", "MH"])  # SingleFam, Condo, PUD, ManufHousing
_RACE = np.array(["White", "Black", "Hispanic", "Asian", "Other"])


def _origination(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    n = int(cfg.generator.n_loans)

    # Vintage cohorts: monthly origination dates between configured bounds
    vint = pd.period_range(cfg.generator.vintage_start, cfg.generator.vintage_end, freq="M")
    orig_period = rng.choice(vint, size=n)
    orig_period = pd.PeriodIndex(orig_period, freq="M")

    # Borrower / loan attributes with realistic mortgage distributions
    # Protected attribute, drawn first so feature distributions can carry the
    # kind of *historical* disparity real fair-lending analysis exists to catch.
    race = rng.choice(_RACE, n, p=[0.62, 0.13, 0.15, 0.07, 0.03])
    # Group offsets on neutral features (NOT endorsed — a synthetic proxy of
    # documented socioeconomic gaps). The model never sees race; these offsets
    # are the *only* pathway by which neutral features can proxy for it, which
    # is exactly the disparate-impact mechanism the fairness screen surfaces.
    fico_shift = pd.Series(race).map(
        {"White": 0, "Asian": 8, "Black": -28, "Hispanic": -18, "Other": -6}).to_numpy()
    dti_shift = pd.Series(race).map(
        {"White": 0, "Asian": -1, "Black": 3.0, "Hispanic": 2.0, "Other": 1.0}).to_numpy()

    fico = np.clip(rng.normal(735, 45, n) + fico_shift, 580, 830).round()
    dti = np.clip(rng.normal(34, 9, n) + dti_shift, 5, 60).round(1)
    ltv = np.clip(rng.normal(76, 13, n), 35, 97).round()
    orig_upb = np.clip(rng.lognormal(mean=12.3, sigma=0.45, size=n), 40_000, 1_200_000)
    orig_upb = (orig_upb / 1000).round() * 1000
    orig_loan_term = rng.choice([180, 240, 360], size=n, p=[0.18, 0.07, 0.75])

    # Rate rises with risk: low FICO / high LTV pay more (used later by LGD too)
    rate = (3.6 + (760 - fico) * 0.004 + (ltv - 75) * 0.012
            + rng.normal(0, 0.35, n))
    rate = np.clip(rate, 2.4, 8.5).round(3)

    occupancy = rng.choice(_OCCUPANCY, n, p=[0.86, 0.10, 0.04])
    purpose = rng.choice(_PURPOSE, n, p=[0.55, 0.20, 0.25])
    prop = rng.choice(_PROPERTY, n, p=[0.70, 0.14, 0.13, 0.03])
    fthb = rng.choice(["Y", "N"], n, p=[0.22, 0.78])

    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "vintage": orig_period.astype(str),
        "orig_date": orig_period.to_timestamp(),
        "fico": fico.astype(int),
        "dti": dti,
        "ltv": ltv.astype(int),
        "orig_interest_rate": rate,
        "orig_upb": orig_upb.astype(int),
        "orig_loan_term": orig_loan_term.astype(int),
        "occupancy_status": occupancy,
        "loan_purpose": purpose,
        "property_type": prop,
        "first_time_homebuyer": fthb,
        "borrower_race": race,
    })
    return df


def _default_probability(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Latent 12-month PD as a logit of the real risk drivers."""
    rng = np.random.default_rng(int(cfg.run.seed) + 7)  # independent noise stream
    z = (
        -0.013 * (df["fico"].to_numpy() - 735)
        + 0.035 * (df["dti"].to_numpy() - 34)
        + 0.030 * (df["ltv"].to_numpy() - 76)
        + 0.18 * (df["orig_interest_rate"].to_numpy() - 4.5)
        + np.where(df["occupancy_status"].to_numpy() == "I", 0.45, 0.0)
        + np.where(df["loan_purpose"].to_numpy() == "C", 0.30, 0.0)
        + np.where(df["property_type"].to_numpy() == "MH", 0.40, 0.0)
        + np.where(df["first_time_homebuyer"].to_numpy() == "Y", 0.15, 0.0)
        # Non-monotonic DTI risk (U-shape): both very low and very high DTI carry
        # extra risk. A monotonic WoE scorecard structurally cannot capture this;
        # the GBM challenger can -> an honest source of challenger edge.
        + 0.035 * ((df["dti"].to_numpy() - 30) / 8.0) ** 2
        # Compounding (super-additive) interactions the monotonic scorecard
        # under-fits but the GBM captures -> an honest source of challenger edge.
        + np.where((df["occupancy_status"].to_numpy() == "I")
                   & (df["ltv"].to_numpy() > 80), 0.80, 0.0)
        + np.where((df["loan_purpose"].to_numpy() == "C")
                   & (df["ltv"].to_numpy() > 85), 0.55, 0.0)
        # Unexplained borrower heterogeneity: keeps feature IVs and Gini in a
        # realistic mortgage range (no single feature dominates / leaks).
        + rng.normal(0, 1.1, len(df))
    )
    # Solve intercept so the population mean PD matches the configured base rate
    target = float(cfg.generator.base_default_rate)
    lo, hi = -12.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1.0 / (1.0 + np.exp(-(mid + z)))
        if p.mean() > target:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    return 1.0 / (1.0 + np.exp(-(intercept + z)))


def _performance(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Monthly performance panel with delinquency march + loss on disposition."""
    horizon = int(cfg.generator.perf_horizon_months)
    obs_window = int(cfg.target.observation_horizon_months)
    n = len(df)

    pd_12 = _default_probability(df, cfg)
    defaults = rng.random(n) < pd_12

    # Default month within the forward observation window (skewed later in window)
    default_month = np.where(
        defaults,
        np.clip((rng.beta(2.2, 2.0, n) * obs_window).astype(int) + 1, 1, obs_window),
        0,
    )

    # LGD severity: higher LTV -> thinner equity cushion -> larger loss
    base_recovery = float(cfg.generator.recovery_mean)
    recovery = np.clip(
        base_recovery + (75 - df["ltv"].to_numpy()) * 0.006 + rng.normal(0, 0.12, n),
        0.05, 0.98,
    )
    lgd = 1.0 - recovery  # fraction of EAD lost

    rows = []
    orig_period = pd.PeriodIndex(df["vintage"], freq="M")
    upb0 = df["orig_upb"].to_numpy().astype(float)
    rate_m = df["orig_interest_rate"].to_numpy() / 100 / 12
    term = df["orig_loan_term"].to_numpy()
    loan_ids = df["loan_id"].to_numpy()

    for i in range(n):
        per = orig_period[i]
        bal = upb0[i]
        # level monthly payment (annuity) for amortisation
        r, T = rate_m[i], term[i]
        pmt = bal * r / (1 - (1 + r) ** (-T)) if r > 0 else bal / T
        dm = default_month[i]
        disposed = False
        for age in range(1, horizon + 1):
            if disposed:
                break
            # amortise principal
            interest = bal * r
            bal = max(bal - (pmt - interest), 0.0)

            zb = pd.NA
            net_loss = np.nan
            if dm and age >= dm:
                months_delinq = age - dm + 3  # entered 90+ at the default month
                dpd_status = str(min(months_delinq, 6))
                if age >= dm + 4:  # disposition ~4 months after default -> REO + loss
                    zb = ZB_REO
                    net_loss = round(bal * lgd[i], 2)
                    disposed = True
            else:
                # healthy loan: usually current, rare transient 30dpd that cures
                if rng.random() < 0.015:
                    dpd_status = "1"
                else:
                    dpd_status = "0"
                # small share of healthy loans prepay and leave the panel
                if not dm and age >= 6 and rng.random() < 0.012:
                    zb = ZB_PREPAID
                    disposed = True

            rows.append((
                loan_ids[i], str(per + age), age,
                round(bal, 2), dpd_status, zb, net_loss,
            ))

    return pd.DataFrame(rows, columns=PERF_SCHEMA)


def generate(cfg: Config | None = None) -> dict[str, Path]:
    """Generate origination + performance panels, write to Bronze, return paths."""
    cfg = cfg or load_config()
    set_global_seed()
    rng = np.random.default_rng(cfg.run.seed)

    orig = _origination(cfg, rng)
    perf = _performance(orig, cfg, rng)

    bronze = cfg.path("bronze")
    (bronze / "origination").mkdir(parents=True, exist_ok=True)
    (bronze / "performance").mkdir(parents=True, exist_ok=True)

    # Partition by vintage YEAR, mirroring Freddie's per-period file layout
    orig["_vyear"] = orig["vintage"].str.slice(0, 4)
    for yr, part in orig.groupby("_vyear"):
        part.drop(columns="_vyear").to_parquet(
            bronze / "origination" / f"orig_{yr}.parquet", index=False)
    orig = orig.drop(columns="_vyear")

    perf = perf.merge(orig[["loan_id", "vintage"]], on="loan_id", how="left")
    perf["_vyear"] = perf["vintage"].str.slice(0, 4)
    for yr, part in perf.groupby("_vyear"):
        part.drop(columns=["_vyear", "vintage"]).to_parquet(
            bronze / "performance" / f"perf_{yr}.parquet", index=False)

    n_def = perf["zero_balance_code"].eq(ZB_REO).sum()
    print(f"[generate] {len(orig):,} loans, {len(perf):,} performance rows, "
          f"{n_def:,} REO dispositions (~{n_def / len(orig):.2%} of book)")
    return {"origination": bronze / "origination", "performance": bronze / "performance"}


if __name__ == "__main__":
    generate()
