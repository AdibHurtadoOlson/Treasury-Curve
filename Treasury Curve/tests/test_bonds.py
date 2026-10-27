import numpy as np
import pytest
from fiworkbench.bonds import cashflows, price_bond, bond_risk, make_curve_function, key_rate_durations

PARAMS = {"beta0": 0.04, "beta1": -0.01, "beta2": 0.01, "tau": 2.0}


def test_cashflows_sum_principal_and_coupons():
    times, flows = cashflows(1000, 0.05, 2.0, 2)
    assert len(times) == 4
    assert flows[-1] == pytest.approx(1025.0)
    assert flows[:-1].sum() == pytest.approx(75.0)


def test_bond_price_falls_when_discount_curve_shifts_up():
    base = make_curve_function(PARAMS)
    up = make_curve_function(PARAMS, lambda t: np.full_like(np.asarray(t, dtype=float), 0.01))
    p0 = price_bond(1000, 0.04, 5.0, base)
    p1 = price_bond(1000, 0.04, 5.0, up)
    assert p1 < p0


def test_duration_and_convexity_positive_for_plain_vanilla_bond():
    risk = bond_risk(1000, 0.04, 5.0, make_curve_function(PARAMS))
    assert risk["price"] > 0
    assert risk["effective_duration_years"] > 0
    assert risk["convexity"] > 0
    assert risk["dv01_for_face_value"] > 0


def test_stub_period_cashflows_end_at_maturity():
    times, flows = cashflows(100, 0.06, 2.2, 2)
    assert times[-1] == pytest.approx(2.2)
    assert times[0] == pytest.approx(0.5)
    assert times[-2] == pytest.approx(2.0)
    assert flows[-2] == pytest.approx(3.0)
    assert flows[-1] == pytest.approx(101.2)


def test_key_rate_durations_have_expected_tenors_and_are_nonnegative_mostly():
    krd = key_rate_durations(1000, 0.04, 5.0, PARAMS)
    assert "DGS5" in krd
    assert len(krd) == 11
    assert np.isfinite(list(krd.values())).all()
