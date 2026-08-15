"""
data/download_eia_demand.py — Download CAISO hourly demand 2021-2024 from EIA API.

This replaces the LMP price approach. Demand data is:
- Complete (no gaps, no rate limiting issues)
- Free (your EIA key works)
- 4 full years including September 2022 heat wave
- Directly relevant to community vulnerability (demand spikes → outage risk)

Research question stays the same, just "demand spike" instead of "price spike":
  Do conformal prediction intervals for CA grid demand spikes exhibit
  systematically lower coverage during extreme heat events in high-burden communities?

Usage:
    cd ~/Desktop/Energy-Research
    python data/download_eia_demand.py

Takes ~5 minutes. Saves data/raw/eia_demand_YYYY.csv for each year.
"""

import requests
import pandas as pd
import time
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

import os
API_KEY = os.environ.get("EIA_API_KEY", "")
if not API_KEY:
    raise SystemExit(
        "Set your EIA API key first (free at eia.gov/opendata/register.php):\n"
        "  export EIA_API_KEY=your_key_here"
    )BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# CISO = California ISO (CAISO)
# Types available: D=Demand, DF=Demand Forecast, G=Generation, TI=Total Interchange
RESPONDENT = "CISO"
DATA_TYPES  = ["D", "DF"]   # Demand + Demand Forecast (both useful as features)


def fetch_eia_chunk(start: str, end: str, data_type: str,
                    offset: int = 0, length: int = 5000) -> list:
    """Fetch one page of EIA data."""
    params = {
        "api_key":              API_KEY,
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": RESPONDENT,
        "facets[type][]":       data_type,
        "start":                start,
        "end":                  end,
        "sort[0][column]":      "period",
        "sort[0][direction]":   "asc",
        "length":               length,
        "offset":               offset,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return result["response"]["data"], int(result["response"]["total"])


def fetch_full_year(year: int, data_type: str) -> pd.DataFrame:
    """Fetch all hourly data for one year and one type, handling pagination."""
    start = f"{year}-01-01T00"
    end   = f"{year}-12-31T23"

    all_rows = []
    offset   = 0
    length   = 5000

    while True:
        rows, total = fetch_eia_chunk(start, end, data_type, offset, length)
        all_rows.extend(rows)
        offset += length
        if offset >= total or len(rows) == 0:
            break
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    return df


def process_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize EIA demand dataframe."""
    df = df.copy()
    # EIA period format: "2021-01-01T00" → datetime
    df["datetime"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["value"]    = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "type-name", "value"]]


def main():
    print("=" * 55)
    print("EIA CAISO Demand Downloader")
    print(f"Respondent: {RESPONDENT} (California ISO)")
    print(f"Types: {DATA_TYPES}")
    print(f"Years: 2021–2024")
    print("=" * 55)

    for year in range(2021, 2025):
        frames = {}

        for dtype in DATA_TYPES:
            print(f"\n  Fetching {year} — {dtype}...", end=" ", flush=True)
            try:
                df = fetch_full_year(year, dtype)
                df = process_demand_df(df)
                frames[dtype] = df
                print(f"{len(df):,} rows ✓")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        if not frames:
            print(f"  ✗ No data for {year}, skipping")
            continue

        # Pivot to wide format: one row per hour, columns = demand + forecast
        demand_df = frames.get("D")
        if demand_df is None:
            continue

        out = demand_df.rename(columns={"value": "demand_mw"}).drop(columns=["type-name"])

        if "DF" in frames:
            forecast_df = frames["DF"].rename(columns={"value": "demand_forecast_mw"})
            forecast_df = forecast_df.drop(columns=["type-name"])
            out = pd.merge(out, forecast_df, on="datetime", how="left")

        # Add derived features
        out["demand_error"]   = out["demand_mw"] - out.get("demand_forecast_mw", out["demand_mw"])
        out["hour"]           = out["datetime"].dt.hour
        out["dow"]            = out["datetime"].dt.dayofweek
        out["month"]          = out["datetime"].dt.month
        out["year"]           = out["datetime"].dt.year
        out["is_weekend"]     = (out["dow"] >= 5).astype(int)

        # Rolling stats
        out["demand_roll24_mean"] = out["demand_mw"].rolling(24).mean()
        out["demand_roll24_std"]  = out["demand_mw"].rolling(24).std()
        out["demand_zscore"]      = (
            (out["demand_mw"] - out["demand_roll24_mean"]) /
            (out["demand_roll24_std"] + 1e-9)
        )

        # Spike label: top 5% of demand within year
        threshold = out["demand_mw"].quantile(0.95)
        out["spike"] = (out["demand_mw"] >= threshold).astype(int)

        # Save
        out_path = RAW_DIR / f"eia_demand_{year}.csv"
        out.to_csv(out_path, index=False)

        print(f"\n  ✓ {year} saved: {len(out):,} rows → {out_path.name}")
        print(f"    Demand range: {out['demand_mw'].min():.0f}–{out['demand_mw'].max():.0f} MW")
        print(f"    Spike rate: {out['spike'].mean():.1%} (target ~5%)")
        print(f"    Missing hours: {out['demand_mw'].isna().sum()}")

        time.sleep(1)

    print("\n" + "=" * 55)
    print("Download complete.")
    print("Next: python data/load_demand.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
