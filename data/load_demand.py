"""
data/load_demand.py — Build research dataset from EIA demand + Open-Meteo weather.

Run after download_eia_demand.py has saved data/raw/eia_demand_YYYY.csv

Usage:
    python data/load_demand.py
    → saves data/processed/merged_demand_hourly.parquet
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime

ROOT          = Path(__file__).parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

WEATHER_LOCATIONS = {
    "san_francisco": (37.77, -122.42),
    "los_angeles":   (34.05, -118.24),
    "sacramento":    (38.58, -121.49),
    "fresno":        (36.75, -119.77),
}

WEATHER_WEIGHTS = {
    "los_angeles": 0.40, "sacramento": 0.25,
    "san_francisco": 0.20, "fresno": 0.15
}


# ── Load EIA demand CSVs ──────────────────────────────────────────────────────

def load_eia_demand() -> pd.DataFrame:
    frames = []
    for year in range(2021, 2025):
        path = RAW_DIR / f"eia_demand_{year}.csv"
        if not path.exists():
            print(f"  ⚠ Missing: {path.name} — run download_eia_demand.py first")
            continue
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        frames.append(df)
        print(f"  ✓ {year}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError("No EIA demand files found. Run download_eia_demand.py")

    combined = pd.concat(frames, ignore_index=True).sort_values("datetime")
    print(f"  → Total: {len(combined):,} rows | "
          f"{combined.datetime.min().date()} to {combined.datetime.max().date()}")
    return combined


# ── Fetch weather ─────────────────────────────────────────────────────────────

def fetch_weather(cache_path: Path = None) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        print("  → Loading cached weather")
        return pd.read_parquet(cache_path)

    print("  Fetching Open-Meteo weather for 4 CA locations...")
    dfs = []
    for name, (lat, lon) in WEATHER_LOCATIONS.items():
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,windspeed_10m,shortwave_radiation",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "start_date": "2021-01-01",
            "end_date":   "2024-12-31",
            "timezone": "UTC",
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params=params, timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "datetime":            pd.to_datetime(h["time"], utc=True),
            f"temp_f_{name}":      h["temperature_2m"],
            f"wind_mph_{name}":    h["windspeed_10m"],
            f"solar_rad_{name}":   h["shortwave_radiation"],
        })
        dfs.append(df)
        print(f"    ✓ {name}: {len(df):,} rows")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="datetime", how="inner")

    # Population-weighted composite
    merged["temp_f"]    = sum(merged[f"temp_f_{n}"] * w
                              for n, w in WEATHER_WEIGHTS.items())
    merged["wind_mph"]  = sum(merged[f"wind_mph_{n}"] * w
                              for n, w in WEATHER_WEIGHTS.items())
    merged["solar_rad"] = sum(merged[f"solar_rad_{n}"] * w
                              for n, w in WEATHER_WEIGHTS.items())
    merged["temp_f_max"] = merged[[f"temp_f_{n}"
                                   for n in WEATHER_WEIGHTS]].max(axis=1)

    if cache_path:
        merged.to_parquet(cache_path, index=False)
    return merged


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # Temporal
    df["hour"]       = df["datetime"].dt.hour
    df["dow"]        = df["datetime"].dt.dayofweek
    df["month"]      = df["datetime"].dt.month
    df["year"]       = df["datetime"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_summer"]  = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Weather features
    df["temp_f_sq"]        = df["temp_f"] ** 2
    df["heat_stress"]      = (df["temp_f"] > 95).astype(int)
    df["extreme_heat"]     = (df["temp_f"] > 105).astype(int)
    df["temp_x_peak_hour"] = df["temp_f"] * df["hour"].isin(range(17, 22)).astype(int)

    # Lagged demand features
    df["demand_lag1h"]       = df["demand_mw"].shift(1)
    df["demand_lag24h"]      = df["demand_mw"].shift(24)
    df["demand_lag168h"]     = df["demand_mw"].shift(168)
    df["demand_roll24_mean"] = df["demand_mw"].shift(1).rolling(24).mean()
    df["demand_roll24_std"]  = df["demand_mw"].shift(1).rolling(24).std()
    df["demand_zscore"]      = (
        (df["demand_mw"] - df["demand_roll24_mean"]) /
        (df["demand_roll24_std"] + 1e-9)
    )

    # Spike label: year-normalized top 5% demand
    df["spike"] = 0
    for year in df["year"].unique():
        mask = df["year"] == year
        threshold = df.loc[mask, "demand_mw"].quantile(0.95)
        df.loc[mask & (df["demand_mw"] >= threshold), "spike"] = 1
    print(f"  → Spike rate: {df['spike'].mean():.1%} (target ~5%)")

    # Heat event flag (3+ consecutive days > 95°F)
    daily_max = df.groupby(df["datetime"].dt.date)["temp_f"].max()
    hot_days  = set(daily_max[daily_max > 95].index)
    df["heat_event"] = df["datetime"].apply(
        lambda dt: int(all(
            (dt.date() - pd.Timedelta(days=i)) in hot_days
            for i in range(3)
        ))
    )
    print(f"  → Heat event hours: {df['heat_event'].sum():,} ({df['heat_event'].mean():.1%})")

    # Temperature regime bins
    df["temp_regime"] = pd.cut(
        df["temp_f"],
        bins=[-np.inf, 75, 90, 100, np.inf],
        labels=["<75°F", "75-90°F", "90-100°F", ">100°F"]
    )

    # September 2022 heat wave flag
    df["sep2022_heatwave"] = (
        (df["year"] == 2022) &
        (df["month"] == 9) &
        (df["datetime"].dt.day.between(5, 9))
    ).astype(int)
    print(f"  → Sep 2022 heat wave hours: {df['sep2022_heatwave'].sum()}")

    return df


def get_splits(df: pd.DataFrame) -> dict:
    train = df[df["year"].isin([2021, 2022])].copy()
    calib = df[df["year"] == 2023].copy()
    test  = df[df["year"] == 2024].copy()
    print(f"\n  Splits:")
    print(f"  Train (2021-22): {len(train):,} | spike={train['spike'].mean():.1%}")
    print(f"  Calib (2023):    {len(calib):,} | spike={calib['spike'].mean():.1%}")
    print(f"  Test  (2024):    {len(test):,}  | spike={test['spike'].mean():.1%}")
    return {"train": train, "calib": calib, "test": test}


def build_dataset() -> pd.DataFrame:
    print("=" * 55)
    print("Building demand research dataset")
    print("=" * 55)

    print("\n[1/4] Loading EIA demand data...")
    demand = load_eia_demand()

    print("\n[2/4] Loading weather data...")
    weather_cache = PROCESSED_DIR / "weather_cache.parquet"
    weather = fetch_weather(cache_path=weather_cache)

    print("\n[3/4] Merging...")
    df = pd.merge(demand, weather, on="datetime", how="inner")
    print(f"  → Merged: {len(df):,} rows")

    print("\n[4/4] Building features...")
    df = build_features(df)

    out = PROCESSED_DIR / "merged_demand_hourly.parquet"
    df.to_parquet(out, index=False)

    # Write manifest
    manifest = {
        "created_at":   datetime.utcnow().isoformat(),
        "random_seed":  RANDOM_SEED,
        "data_source":  "EIA Open Data API v2 — CAISO hourly demand (respondent=CISO, type=D)",
        "eia_api_url":  "https://api.eia.gov/v2/electricity/rto/region-data/data/",
        "weather_source": "Open-Meteo Archive API",
        "date_range":   {"start": "2021-01-01", "end": "2024-12-31"},
        "spike_definition": "Top 5% hourly demand within each calendar year",
        "train_years":  [2021, 2022],
        "calib_year":   2023,
        "test_year":    2024,
        "key_event":    "September 5-9 2022 CA heat wave (sep2022_heatwave flag)",
        "note": "Switched from CAISO LMP prices to EIA demand data due to CAISO API historical limits. Demand spikes are directly relevant to community vulnerability (outage risk).",
    }
    import json
    with open(ROOT / "data" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved: {out}")
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    return df


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "merged_demand_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError("Run: python data/load_demand.py")
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_dataset()
    splits = get_splits(df)
    print("\nSample:")
    cols = ["datetime", "demand_mw", "spike", "temp_f",
            "heat_stress", "heat_event", "sep2022_heatwave", "temp_regime"]
    print(df[cols].dropna().tail(5).to_string(index=False))
