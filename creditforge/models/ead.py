"""EAD — Exposure at Default (Part 3.4).

For a fully-drawn, fully-amortizing mortgage EAD is essentially the outstanding
balance, so the general Credit-Conversion-Factor framework collapses to
EAD = drawn_balance + CCF · undrawn. With no undrawn commitment we use the
origination UPB scaled by the configured CCF (1.0). The hook is left explicit
so a revolving/undrawn product can plug in a fitted CCF model later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditforge.config import Config, load_config


def estimate_ead(df: pd.DataFrame, cfg: Config | None = None) -> np.ndarray:
    """EAD per loan. Mortgage: CCF · origination UPB (no undrawn commitment)."""
    cfg = cfg or load_config()
    ccf = float(cfg.ead.ccf)
    return (df["orig_upb"].to_numpy(dtype=float) * ccf)
