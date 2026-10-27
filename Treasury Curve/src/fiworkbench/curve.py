"""Nelson–Siegel yield curve fitting and smooth curve evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .data import TENORS


def nelson_siegel(maturity_years: np.ndarray | float, beta0: float, beta1: float,
                  beta2: float, tau: float) -> np.ndarray:
    """Return the four-parameter Nelson–Siegel yield for maturities in years."""
    t = np.asarray(maturity_years, dtype=float)
    if tau <= 0 or np.any(t < 0):
        raise ValueError("tau must be positive and maturities cannot be negative.")
    x = np.maximum(t, 1e-10) / tau
    loading1 = (1.0 - np.exp(-x)) / x
    loading2 = loading1 - np.exp(-x)
    return beta0 + beta1 * loading1 + beta2 * loading2


def fit_nelson_siegel(observed_yields: pd.Series, min_points: int = 6) -> dict[str, float]:
    """Fit a smooth curve to available CMT maturity/yield observations."""
    pairs = [(TENORS[col], val) for col, val in observed_yields.items()
             if col in TENORS and pd.notna(val) and np.isfinite(float(val))]
    if len(pairs) < min_points:
        raise ValueError(f"Need at least {min_points} valid tenor observations; got {len(pairs)}.")
    t = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    # Bounds discourage implausibly explosive curve parameters but still allow
    # negative yield environments. tau is constrained to a positive range.
    initial = np.array([float(np.median(y)), float(y[0] - y[-1]), 0.01, 2.0])
    fit = least_squares(lambda p: nelson_siegel(t, *p) - y, initial,
                        bounds=([-0.10, -0.30, -0.30, 0.05], [0.30, 0.30, 0.30, 50.0]),
                        loss="soft_l1", f_scale=0.002, max_nfev=20000)
    b0, b1, b2, tau = map(float, fit.x)
    residuals = nelson_siegel(t, b0, b1, b2, tau) - y
    return {"beta0": b0, "beta1": b1, "beta2": b2, "tau": tau,
            "n_observations": int(len(t)), "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "max_abs_residual": float(np.max(np.abs(residuals))), "success": bool(fit.success)}


def fit_latest_curve(curve_history: pd.DataFrame) -> tuple[pd.Timestamp, dict[str, float], pd.DataFrame]:
    """Fit the most recent date with enough tenor observations and return residual diagnostics."""
    for dt, row in curve_history.sort_index(ascending=False).iterrows():
        try:
            params = fit_nelson_siegel(row)
            tenors = np.array([TENORS[c] for c in row.index if c in TENORS and pd.notna(row[c])])
            cols = [c for c in row.index if c in TENORS and pd.notna(row[c])]
            observed = np.array([float(row[c]) for c in cols])
            fitted = nelson_siegel(tenors, params["beta0"], params["beta1"], params["beta2"], params["tau"])
            residuals = pd.DataFrame({"tenor": cols, "maturity_years": tenors,
                                      "observed_yield": observed, "fitted_yield": fitted,
                                      "residual": fitted - observed})
            return pd.Timestamp(dt), params, residuals
        except ValueError:
            continue
    raise ValueError("No observation date has enough nonmissing Treasury tenors to fit a curve.")
