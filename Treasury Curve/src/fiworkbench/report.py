"""Turn a fitted FRED curve and bond configuration into an analyst risk pack."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .curve import fit_latest_curve, nelson_siegel
from .bonds import make_curve_function, price_bond, bond_risk, key_rate_durations, scenario_shocks
from .data import TENORS

SCENARIOS = ["parallel_up_100bp", "parallel_down_100bp", "bear_steepener", "bull_steepener", "belly_selloff"]


def run_analytics(curves: pd.DataFrame, bond_config: pd.DataFrame, outdir: Path, source: str) -> dict:
    """Fit the latest curve, value the configured positions, and export scenarios."""
    outdir.mkdir(parents=True, exist_ok=True)
    curves = curves.sort_index()
    curves.to_csv(outdir / "treasury_curve_history.csv", index_label="Date")
    valuation_date, params, residuals = fit_latest_curve(curves)
    residuals.to_csv(outdir / "latest_curve_fit.csv", index=False)
    curve_grid = np.geomspace(1/12, 30, 300)
    fit_grid = nelson_siegel(curve_grid, params["beta0"], params["beta1"], params["beta2"], params["tau"])
    (outdir / "curve_fit_diagnostics.json").write_text(json.dumps({"valuation_date": str(valuation_date.date()), **params}, indent=2), encoding="utf-8")
    base_curve = make_curve_function(params)
    bond_rows, key_rows = [], []
    for row in bond_config.itertuples(index=False):
        risk = bond_risk(float(row.face_value), float(row.coupon_rate), float(row.maturity_years), base_curve, int(row.frequency))
        market_value = risk["price"] * float(row.quantity)
        bond_rows.append({"bond_id": row.bond_id, "face_value": row.face_value, "coupon_rate": row.coupon_rate,
                          "maturity_years": row.maturity_years, "frequency": row.frequency, "quantity": row.quantity,
                          **risk, "market_value": market_value,
                          "position_dv01": risk["dv01_for_face_value"] * float(row.quantity)})
        krd = key_rate_durations(float(row.face_value), float(row.coupon_rate), float(row.maturity_years), params, int(row.frequency))
        key_rows.append({"bond_id": row.bond_id, **{k: v for k, v in krd.items()}})
    bonds = pd.DataFrame(bond_rows)
    bonds.to_csv(outdir / "bond_valuation.csv", index=False)
    key_rate_frame = pd.DataFrame(key_rows).set_index("bond_id")
    key_rate_frame.to_csv(outdir / "key_rate_duration.csv")
    total_mv = float(bonds["market_value"].sum())
    total_dv01 = float(bonds["position_dv01"].sum())
    # Convert key-rate duration into dollar DV01 using each position's value,
    # then aggregate by tenor for a portfolio-level risk view.
    market_values = bonds.set_index("bond_id")["market_value"].reindex(key_rate_frame.index)
    portfolio_kr_rows = []
    for tenor_col in [c for c in key_rate_frame.columns if c in TENORS]:
        dollar_krd = key_rate_frame[tenor_col] * market_values * 0.0001
        portfolio_kr_rows.append({"tenor": tenor_col,
                                  "portfolio_key_rate_duration": float((key_rate_frame[tenor_col] * market_values).sum() / total_mv),
                                  "portfolio_key_rate_dv01": float(dollar_krd.sum())})
    portfolio_kr = pd.DataFrame(portfolio_kr_rows).set_index("tenor")
    portfolio_kr.to_csv(outdir / "portfolio_key_rate_risk.csv")
    # A shocked curve is the base fitted curve plus a tenor-dependent shift.
    scen_rows = []
    for sc in SCENARIOS:
        value = 0.0
        for row in bond_config.itertuples(index=False):
            shock_curve = make_curve_function(params, lambda t, s=sc: scenario_shocks(t, s))
            value += price_bond(float(row.face_value), float(row.coupon_rate), float(row.maturity_years),
                                shock_curve, int(row.frequency)) * float(row.quantity)
        scen_rows.append({"scenario": sc, "portfolio_value": value,
                          "pnl": value - total_mv, "pnl_pct": value / total_mv - 1 if total_mv else np.nan})
    scenarios = pd.DataFrame(scen_rows).set_index("scenario")
    scenarios.to_csv(outdir / "rate_scenarios.csv")
    summary = {"data_source": source, "valuation_date": str(valuation_date.date()),
               "yield_units": "decimal annual rates internally; source quotes were percentage values",
               "curve_method": "Nelson-Siegel fit to Treasury constant-maturity yield quotes; not an arbitrage-free bootstrapped zero curve",
               "curve_fit": params, "portfolio_market_value_proxy": total_mv,
               "portfolio_position_dv01_per_bp_proxy": total_dv01,
               "weighted_average_effective_duration": float(np.average(bonds["effective_duration_years"], weights=bonds["market_value"])),
               "portfolio_key_rate_dv01": portfolio_kr["portfolio_key_rate_dv01"].to_dict(),
               "scenario_pnl": scenarios["pnl"].to_dict(),
               "notes": ["Bond cash flows are stylized configurable proxies, not live CUSIP quotes.",
                         "Curve-based valuations use continuous-compounding fitted CMT yields as a practical approximation.",
                         "Settlement, accrued interest, coupon calendars, and day-count conventions are not modeled."]}
    (outdir / "fixed_income_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    latest = curves.loc[valuation_date]
    cols = [c for c in latest.index if c in TENORS and pd.notna(latest[c])]
    axes[0, 0].plot([TENORS[c] for c in cols], [latest[c] * 100 for c in cols], "o", label="FRED CMT quotes")
    axes[0, 0].plot(curve_grid, fit_grid * 100, label="Nelson–Siegel fit")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_title(f"Treasury CMT curve — {valuation_date.date()}")
    axes[0, 0].set_xlabel("Maturity (years, log scale)"); axes[0, 0].set_ylabel("Yield (%)"); axes[0, 0].legend()
    axes[0, 1].bar(bonds["bond_id"], bonds["market_value"]); axes[0, 1].set_title("Position market-value proxy"); axes[0, 1].tick_params(axis="x", rotation=20)
    axes[1, 0].bar(bonds["bond_id"], bonds["effective_duration_years"]); axes[1, 0].set_title("Effective duration (years)"); axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].bar(scenarios.index, scenarios["pnl"]); axes[1, 1].set_title("Scenario P&L proxy"); axes[1, 1].tick_params(axis="x", rotation=25)
    fig.savefig(outdir / "fixed_income_dashboard.png", dpi=160); plt.close(fig)
    return summary
