"""Expected Loss — the deterministic combination EL = PD · LGD · EAD (Part 3.4).

The models *supply* PD/LGD/EAD; this module just does the arithmetic, so it is
pure, deterministic, and unit-tested (Part 10.6). Per-loan EL and portfolio EL.
"""
from __future__ import annotations

import numpy as np


def expected_loss(pd_hat, lgd_hat, ead) -> np.ndarray:
    """Per-loan Expected Loss. All inputs broadcast to the same shape."""
    pd_hat = np.asarray(pd_hat, float)
    lgd_hat = np.asarray(lgd_hat, float)
    ead = np.asarray(ead, float)
    if np.any((pd_hat < 0) | (pd_hat > 1)):
        raise ValueError("PD must be in [0, 1]")
    if np.any((lgd_hat < 0) | (lgd_hat > 1)):
        raise ValueError("LGD must be in [0, 1]")
    if np.any(ead < 0):
        raise ValueError("EAD must be non-negative")
    return pd_hat * lgd_hat * ead


def portfolio_el(pd_hat, lgd_hat, ead) -> float:
    return float(expected_loss(pd_hat, lgd_hat, ead).sum())
