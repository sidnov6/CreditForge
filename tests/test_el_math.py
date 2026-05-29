"""Unit tests for the deterministic Expected-Loss arithmetic (Part 10.6)."""
import numpy as np
import pytest

from creditforge.models.el import expected_loss, portfolio_el


def test_el_is_product():
    el = expected_loss([0.1, 0.5], [0.4, 0.8], [100_000, 200_000])
    np.testing.assert_allclose(el, [4000.0, 80_000.0])


def test_portfolio_el_sums():
    assert portfolio_el([0.1, 0.2], [0.5, 0.5], [1000, 1000]) == pytest.approx(150.0)


def test_scalar_inputs_ok():
    assert float(expected_loss(0.05, 0.45, 250_000)) == pytest.approx(5625.0)


def test_zero_pd_zero_loss():
    assert portfolio_el([0.0, 0.0], [0.9, 0.9], [1e6, 1e6]) == 0.0


@pytest.mark.parametrize("pd_, lgd, ead", [
    (-0.01, 0.5, 100), (1.2, 0.5, 100),     # PD out of range
    (0.1, -0.1, 100), (0.1, 1.5, 100),      # LGD out of range
    (0.1, 0.5, -5),                          # negative EAD
])
def test_invalid_inputs_raise(pd_, lgd, ead):
    with pytest.raises(ValueError):
        expected_loss(pd_, lgd, ead)
