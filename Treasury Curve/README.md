# Treasury Curve & Fixed-Income Risk Workbench

A market-data fixed-income workflow using actual U.S. Treasury constant-maturity yields from the Federal Reserve Bank of St. Louis FRED service. The program downloads the daily curve, fits a Nelson–Siegel term-structure approximation, prices a configurable cash-flow portfolio, computes duration/convexity/DV01 and key-rate risk, and simulates parallel and curve-shape shocks.

## Real data source and key caveat

- **Source:** FRED constant-maturity Treasury series `DGS1MO,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30`.
- The public FRED graph CSV endpoint is used, so no API key is required for the standard download: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=...`.
- Constant-maturity Treasury observations are market yield quotes, not a directly observed zero-coupon spot curve. The Nelson–Siegel fit here is therefore a **smooth approximation to the constant-maturity yield curve**. The bond pricing module discounts cash flows off that fitted curve as a practical approximation; it is not a full institutional Treasury STRIPS bootstrap or a bond-specific dealer quote.
- The default portfolio file contains stylized bond cash flows, not live CUSIPs. Replace it with documented positions if you have a proper bond holdings file.
- Successful downloads are cached. The run refreshes from FRED by default; `--use-cache` is explicit. If a live call fails, the program may use a prior cache only when `--allow-stale-cache` is set, and it records the source date in outputs.

## Capabilities

- Downloads and validates daily Treasury CMT data, including date parsing, missing values, tenor completeness, percentage-to-decimal conversion, and source-date checks.
- Fits a constrained Nelson–Siegel curve with SciPy nonlinear least squares and reports fitting residuals.
- Values fixed-rate bond cash flows using interpolated fitted yields as a continuous-compounding discount proxy.
- Calculates price, Macaulay duration, modified/effective duration, convexity, DV01, and key-rate duration by tenor.
- Runs parallel up/down shifts, bear steepener, bull steepener, and belly/twist scenarios at portfolio level.
- Writes curve, valuation, risk, scenario, and dashboard outputs; includes mathematical and data-validation tests.

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python run.py --start 2015-01-01
```

Run using cached curve observations:

```bash
python run.py --use-cache
```

Edit `config/bond_portfolio.csv` to change face value, coupon, maturity years, coupon frequency, and number of bonds held. Rates are retrieved as percent quotes and normalized into decimal units internally.

## Outputs

`outputs/treasury_curve_history.csv`, `outputs/latest_curve_fit.csv`, `outputs/curve_fit_diagnostics.json`, `outputs/bond_valuation.csv`, `outputs/key_rate_duration.csv`, `outputs/portfolio_key_rate_risk.csv`, `outputs/rate_scenarios.csv`, `outputs/fixed_income_summary.json`, and `outputs/fixed_income_dashboard.png`.

## Extensions for a real analyst workflow

- Bootstrap an arbitrage-consistent discount curve from bill, note and bond prices / coupon rates, then reconcile the fit with tradeable market quotes.
- Add settlement, accrued interest, day-count conventions, business-day calendars, holidays, and actual coupon dates.
- Connect holdings and benchmark data, calculate key-rate and spread DV01, separate Treasury and credit spread risk, and compare scenario results to risk limits.
- For production use, obtain licensed data, independent price verification, controlled model governance, and documented curve construction.
