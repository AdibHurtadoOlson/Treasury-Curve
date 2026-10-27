"""Stylized fixed-rate bond valuation and curve-sensitivity calculations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .curve import nelson_siegel
from .data import TENORS


def make_curve_function(params: dict[str, float], shock=None):
    """Create a callable continuous-compounding yield proxy, optionally shocked."""
    def curve(t):
        t_arr = np.asarray(t, dtype=float)
        y = nelson_siegel(t_arr, params["beta0"], params["beta1"], params["beta2"], params["tau"])
        if shock is not None:
            y = y + shock(t_arr)
        return y
    return curve


def cashflows(face_value: float, coupon_rate: float, maturity_years: float, frequency: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Construct coupon/principal cash flows, allowing a short final stub period."""
    if face_value <= 0 or coupon_rate < 0 or maturity_years <= 0 or frequency not in {1, 2, 4, 12}:
        raise ValueError("Face/maturity must be positive, coupon nonnegative, and frequency in {1,2,4,12}.")
    step = 1.0 / frequency
    n_full = int(np.floor(maturity_years / step + 1e-10))
    times = np.arange(1, n_full + 1, dtype=float) * step
    times = times[times < maturity_years - 1e-9]
    times = np.r_[times, float(maturity_years)]
    periods = np.diff(np.r_[0.0, times])
    flows = face_value * coupon_rate * periods
    flows[-1] += face_value
    return times, flows


def price_bond(face_value: float, coupon_rate: float, maturity_years: float, curve_fn,
               frequency: int = 2) -> float:
    """Present-value coupon and principal flows on the chosen fitted yield curve."""
    times, flows = cashflows(face_value, coupon_rate, maturity_years, frequency)
    rates = np.asarray(curve_fn(times), dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("Curve returned nonfinite discount rates.")
    return float(np.sum(flows * np.exp(-rates * times)))


def bond_risk(face_value: float, coupon_rate: float, maturity_years: float, curve_fn,
              frequency: int = 2, bump_bp: float = 1.0) -> dict[str, float]:
    """Calculate price, cash-flow duration, effective duration, convexity and DV01."""
    times, flows = cashflows(face_value, coupon_rate, maturity_years, frequency)
    base_rates = np.asarray(curve_fn(times), dtype=float)
    discounts = np.exp(-base_rates * times)
    price = float(np.sum(flows * discounts))
    macaulay = float(np.sum(times * flows * discounts) / price)
    bump = bump_bp / 10000.0
    p_up = float(np.sum(flows * np.exp(-(base_rates + bump) * times)))
    p_down = float(np.sum(flows * np.exp(-(base_rates - bump) * times)))
    effective_duration = (p_down - p_up) / (2.0 * price * bump)
    convexity = (p_down + p_up - 2 * price) / (price * bump * bump)
    dv01 = (p_down - p_up) / 2.0 * (1.0 / bump_bp)
    return {"price": price, "macaulay_duration_years": macaulay,
            "effective_duration_years": float(effective_duration), "convexity": float(convexity),
            "dv01_for_face_value": float(dv01), "price_up_1bp": p_up, "price_down_1bp": p_down}


def key_rate_durations(face_value: float, coupon_rate: float, maturity_years: float,
                       params: dict[str, float], frequency: int = 2, bump_bp: float = 1.0) -> dict[str, float]:
    """Estimate localized key-rate duration using triangular tenor-node bumps."""
    tenor_items = sorted(TENORS.items(), key=lambda item: item[1])
    node_years = np.array([v for _, v in tenor_items])
    names = [k for k, _ in tenor_items]
    base_curve = make_curve_function(params)
    base_price = price_bond(face_value, coupon_rate, maturity_years, base_curve, frequency)
    bump = bump_bp / 10000.0
    result = {}
    for name, node in tenor_items:
        # Width reaches halfway to neighboring nodes on either side, so each
        # shift is localized rather than a parallel bump to the whole curve.
        idx = names.index(name)
        left = node - node_years[idx - 1] if idx > 0 else (node_years[1] - node)
        right = node_years[idx + 1] - node if idx < len(names) - 1 else (node - node_years[idx - 1])
        width = max(left, right, 0.1)
        def bump_shape(t, sign):
            arr = np.asarray(t, dtype=float)
            return sign * bump * np.maximum(0.0, 1.0 - np.abs(arr - node) / width)
        p_up = price_bond(face_value, coupon_rate, maturity_years,
                          make_curve_function(params, lambda t, s=1: bump_shape(t, s)), frequency)
        p_down = price_bond(face_value, coupon_rate, maturity_years,
                            make_curve_function(params, lambda t, s=-1: bump_shape(t, s)), frequency)
        result[name] = float((p_down - p_up) / (2.0 * base_price * bump))
    return result


def scenario_shocks(maturity_years: np.ndarray, scenario: str) -> np.ndarray:
    """Return illustrative curve shifts in decimal-rate units for each tenor."""
    t = np.asarray(maturity_years, dtype=float)
    if scenario == "parallel_up_100bp": return np.full_like(t, 0.0100)
    if scenario == "parallel_down_100bp": return np.full_like(t, -0.0100)
    # Short rates move +50 bp and long rates +100 bp in a bear steepener.
    if scenario == "bear_steepener": return 0.005 + 0.005 * np.clip(t / 10.0, 0, 1)
    if scenario == "bull_steepener": return -0.010 + 0.005 * np.clip(t / 10.0, 0, 1)
    if scenario == "belly_selloff": return 0.010 * np.exp(-0.5 * ((t - 5.0) / 3.0) ** 2)
    raise ValueError(f"Unknown scenario {scenario!r}.")
