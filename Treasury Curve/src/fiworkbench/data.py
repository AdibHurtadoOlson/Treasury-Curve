"""FRED daily Treasury constant-maturity yield data loader."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests

TENORS = {"DGS1MO": 1/12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
          "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0,
          "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(TENORS)


def validate_curve_data(frame: pd.DataFrame, min_nonmissing_tenors: int = 6) -> pd.DataFrame:
    """Normalize FRED observations and enforce meaningful daily curve rows.

    Values are stored in decimal annual yield form after normalization.
    Empty strings and FRED's '.' missing marker are treated as missing, never 0%.
    """
    df = frame.copy()
    if "observation_date" in df.columns:
        df = df.set_index("observation_date")
    if "DATE" in df.columns:
        df = df.set_index("DATE")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    if df.index.has_duplicates:
        raise ValueError("FRED curve data contains duplicate observation dates.")
    known = [c for c in TENORS if c in df.columns]
    if len(known) < min_nonmissing_tenors:
        raise ValueError(f"Expected at least {min_nonmissing_tenors} Treasury tenors, received {known}.")
    df = df[known].replace({".": float("nan"), "": float("nan"), "nan": float("nan")})
    for col in known:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    # Quotes outside this broad guardrail are almost certainly data/units errors.
    if ((df[known] < -0.05) | (df[known] > 0.50)).any().any():
        raise ValueError("Yield quote outside broad -5% to 50% validation bounds; inspect source units.")
    df = df.dropna(how="all")
    if len(df) < 10:
        raise ValueError("Too few valid FRED observations to fit a yield curve.")
    df.index.name = "Date"
    return df


def fetch_fred_curve(cache_path: Path, start: str = "2015-01-01", use_cache: bool = False,
                     allow_stale_cache: bool = False) -> tuple[pd.DataFrame, str]:
    """Fetch daily yields from the public FRED CSV endpoint and cache decimal rates.

    The network response quotes yields in percent. The validator converts them
    to decimal form once, and both successful and cached results use decimals.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if use_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f"No curve cache at {cache_path}; run once online first.")
        raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        known = [c for c in TENORS if c in raw.columns]
        if len(known) < 6 or raw.index.has_duplicates:
            raise ValueError("Cached Treasury curve is missing tenors or contains duplicate dates.")
        raw = raw[known].apply(pd.to_numeric, errors="coerce")
        if ((raw < -0.05) | (raw > 0.50)).any().any():
            raise ValueError("Cached Treasury yields failed decimal-unit validation.")
        raw = raw.dropna(how="all").loc[lambda x: x.index >= pd.Timestamp(start)]
        if len(raw) < 10:
            raise ValueError("Cached Treasury curve has fewer than 10 observations in the requested date range.")
        return raw, "FRED cache"
    try:
        from io import StringIO
        headers = {"User-Agent": "quant-finance-portfolio/1.0 (research project)"}
        # Try one multi-series CSV request first for a fast daily refresh.
        response = requests.get(FRED_URL, params={"cosd": start}, headers=headers, timeout=30)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), na_values=[".", ""])
        try:
            # Multi-series CSV is supported by many FRED graph URLs. Validate
            # the actual response rather than assuming the server returned data.
            normalized = validate_curve_data(raw)
        except (ValueError, KeyError):
            # If the endpoint returns only one series or an unexpected payload,
            # retrieve individual two-column CSVs and merge by observation date.
            # This is slower but makes the connector robust to endpoint behavior.
            series_frames = []
            base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
            for series_id in TENORS:
                single = requests.get(base_url, params={"id": series_id, "cosd": start},
                                      headers=headers, timeout=30)
                single.raise_for_status()
                one = pd.read_csv(StringIO(single.text), na_values=[".", ""])
                date_col = "observation_date" if "observation_date" in one.columns else ("DATE" if "DATE" in one.columns else None)
                if date_col is None or series_id not in one.columns:
                    raise ValueError(f"FRED response for {series_id} did not contain date and series columns.")
                one[date_col] = pd.to_datetime(one[date_col], errors="coerce")
                values = pd.Series(pd.to_numeric(one[series_id], errors="coerce").to_numpy(),
                                   index=one[date_col], name=series_id)
                series_frames.append(values)
            raw_joined = pd.concat(series_frames, axis=1)
            raw_joined.index.name = "Date"
            normalized = validate_curve_data(raw_joined)
        normalized = normalized.loc[normalized.index >= pd.Timestamp(start)]
        # Save decimal-normalized values; they are unambiguously named and
        # documented in the cache file header (percent values divided by 100).
        normalized.to_csv(cache_path, index_label="Date")
        return normalized, "Federal Reserve Bank of St. Louis FRED daily Treasury CMT series"
    except Exception as exc:
        if allow_stale_cache and cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()
            known = [c for c in TENORS if c in cached.columns]
            if len(known) < 6 or cached.index.has_duplicates:
                raise RuntimeError("Stale Treasury cache failed tenor/date validation.") from exc
            cached = cached[known].apply(pd.to_numeric, errors="coerce")
            cached = cached.loc[cached.index >= pd.Timestamp(start)].dropna(how="all")
            if len(cached) < 10:
                raise RuntimeError(f"Live FRED download failed and stale cache has fewer than 10 observations after {start}.") from exc
            return cached, f"STALE FRED CACHE (live download failed: {type(exc).__name__})"
        raise RuntimeError(f"Could not retrieve Treasury yields from FRED. Check internet connectivity or use --use-cache after a successful run. Details: {exc}") from exc
