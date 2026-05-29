"""Unit tests for feature transforms, the Silver quality gate, and PSI/CSI."""
import numpy as np
import pandas as pd
import pytest

from creditforge.pipeline.gold import _engineer
from creditforge.pipeline.silver import QUALITY_RANGES
from creditforge.validation.stability import psi, psi_status, csi


def test_engineered_features():
    df = pd.DataFrame({
        "vintage": ["2020-01", "2020-01"],
        "ltv": [80, 90], "dti": [40, 50], "orig_interest_rate": [4.0, 5.0],
    })
    out = _engineer(df)
    assert out["ltv_dti"].tolist() == [32.0, 45.0]            # ltv*dti/100
    assert out["high_ltv"].tolist() == [0, 1]                  # >80 threshold
    # rate_spread is centred within the vintage -> sums to ~0 for the cohort
    assert out["rate_spread"].sum() == pytest.approx(0.0, abs=1e-9)


def test_quality_ranges_are_sane():
    lo, hi = QUALITY_RANGES["fico"]
    assert lo < hi and lo >= 300 and hi <= 850


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 8000)
    small = psi(base, rng.normal(0.2, 1, 8000))
    large = psi(base, rng.normal(1.5, 1, 8000))
    assert 0 <= small < large
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "watch"
    assert psi_status(0.40) == "unstable"


def test_csi_categorical():
    a = pd.Series(["P"] * 80 + ["I"] * 20)
    b = pd.Series(["P"] * 50 + ["I"] * 50)
    assert csi(a, b) > 0.1   # a clear categorical shift
