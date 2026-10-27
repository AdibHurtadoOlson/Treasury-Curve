import numpy as np
import pandas as pd
from fiworkbench.curve import nelson_siegel
from fiworkbench.data import TENORS
from fiworkbench.report import run_analytics

def test_fixed_income_pack_runs_on_curve_fixture(tmp_path):
    rng = np.random.default_rng(44)
    dates = pd.date_range("2025-01-02", periods=15, freq="B")
    columns = list(TENORS)
    mats = np.array([TENORS[c] for c in columns])
    rows = []
    for _ in dates:
        rows.append(nelson_siegel(mats, 0.04, -0.01, 0.008, 2.4) + rng.normal(0, 0.0002, len(mats)))
    curves = pd.DataFrame(rows, index=dates, columns=columns)
    bonds = pd.DataFrame([
        {"bond_id": "A", "face_value": 100000, "coupon_rate": 0.04, "maturity_years": 2.0, "frequency": 2, "quantity": 1},
        {"bond_id": "B", "face_value": 100000, "coupon_rate": 0.045, "maturity_years": 8.0, "frequency": 2, "quantity": 2},
    ])
    summary = run_analytics(curves, bonds, tmp_path, "test fixture")
    assert summary["portfolio_market_value_proxy"] > 0
    assert summary["portfolio_position_dv01_per_bp_proxy"] > 0
    for name in ["bond_valuation.csv", "portfolio_key_rate_risk.csv", "rate_scenarios.csv", "fixed_income_dashboard.png"]:
        assert (tmp_path / name).exists()
