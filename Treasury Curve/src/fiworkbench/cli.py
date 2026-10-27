"""Command-line entry point for Treasury curve and fixed-income risk analytics."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data import fetch_fred_curve, TENORS
from .report import run_analytics

DEFAULT_BONDS = Path("config/bond_portfolio.csv")

def main() -> None:
    p = argparse.ArgumentParser(description="Fit the FRED Treasury CMT curve and value stylized bond positions.")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--allow-stale-cache", action="store_true")
    p.add_argument("--cache", default="data/cache/fred_treasury_cmt.csv")
    p.add_argument("--bonds", default=str(DEFAULT_BONDS))
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args()
    curves, source = fetch_fred_curve(Path(a.cache), a.start, a.use_cache, a.allow_stale_cache)
    bonds = pd.read_csv(a.bonds)
    required = {"bond_id", "face_value", "coupon_rate", "maturity_years", "frequency", "quantity"}
    if not required.issubset(bonds.columns):
        p.error(f"Bond config must contain columns {sorted(required)}")
    if (bonds["face_value"] <= 0).any() or (bonds["quantity"] <= 0).any():
        p.error("face_value and quantity must be positive")
    if (bonds["coupon_rate"] < 0).any() or (bonds["maturity_years"] <= 0).any():
        p.error("coupon_rate must be nonnegative and maturity_years positive")
    summary = run_analytics(curves, bonds, Path(a.outdir), source)
    print(f"Fixed-income risk pack written to {Path(a.outdir).resolve()}")
    print(f"Valuation date: {summary['valuation_date']}; portfolio value proxy: ${summary['portfolio_market_value_proxy']:,.2f}")
    print(f"Position DV01 proxy: ${summary['portfolio_position_dv01_per_bp_proxy']:,.2f} per bp")

if __name__ == "__main__": main()
