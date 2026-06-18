"""
data/load.py — Load and preprocess real CAISO LMP + Open-Meteo weather data.

SETUP INSTRUCTIONS:
1. Download CAISO LMP CSVs from: http://oasis.caiso.com/mrioasis/logon.do
   - Query: PRC_LMP, RTM, node=TH_NP15_GEN-APND
   - Date range: 2021-01-01 to 2024-12-31
   - Save files as: data/raw/caiso_lmp_YYYY.csv (one per year)

2. Weather is fetched automatically via Open-Meteo archive API (free, no key needed)

3. Run: python data/load.py
   → saves data/processed/merged_hourly.parquet
   → saves data/manifest.json

Usage:
    from data.load import load_processed
    df = load_processed()
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, date
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Weather locations (4 CA climate zones) ───────────────────────────────────
WEATHER_LOCATIONS = {
    "san_francisco": (37.77, -122.42),
    "los_angeles":   (34.05, -118.24),
    "sacramento":    (38.58, -121.49),
    "fresno":        (36.75, -119.77),
}

START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"


# ── CAISO LMP Loading ─────────────────────────────────────────────────────────

def load_caiso_csvs() -> pd.DataFrame:
    """
    Load CAISO LMP CSVs from data/raw/.
    Expected filename pattern: caiso_lmp_YYYY.csv
    Columns expected: INTERVALSTARTTIME_GMT, MW (or similar — handles variants)
    """
    frames = []
    for year in range(2021, 2025):
        path = RAW_DIR / f"caiso_lmp_{year}.csv"
        if not path.exists():
            print(f"  ⚠ Missing: {path.name} — run CAISO download first")
            continue
        df = pd.read_csv(path, low_memory=False)

        # Flexible column detection — CAISO CSV headers vary slightly
        time_col = next((c for c in df.columns if "INTERVALSTART" in c.upper()), None)
        price_col = next((c for c in df.columns if c.upper() in ("MW", "LMP", "VALUE")), None)
        if time_col is None or price_col is None:
            print(f"  ⚠ Could not parse columns in {path.name}: {list(df.columns)[:6]}")
            continue

        df = df[[time_col, price_col]].copy()
        df.columns = ["datetime", "lmp"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
        df = df.dropna()
        # Floor to hour and average within hour (5-min → hourly)
        df["datetime"] = df["datetime"].dt.floor("h")
        df = df.groupby("datetime")["lmp"].mean().reset_index()
        frames.append(df)
        print(f"  ✓ Loaded {path.name}: {len(df):,} hourly rows")

    if not frames:
        raise FileNotFoundError(
            "No CAISO CSV files found in data/raw/.\n"
            "Download from: http://oasis.caiso.com/mrioasis/logon.do\n"
            "Query: PRC_LMP | RTM | TH_NP15_GEN-APND | 2021-01-01 to 2024-12-31\n"
            "Save as: data/raw/caiso_lmp_YYYY.csv"
        )

    lmp = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    print(f"  → Total LMP rows: {len(lmp):,} | Range: {lmp.datetime.min()} – {lmp.datetime.max()}")
    return lmp


# ── Weather Fetching ──────────────────────────────────────────────────────────

def fetch_weather_location(name: str, lat: float, lon: float,
                           start: str, end: str) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,shortwave_radiation,precipitation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params, timeout=60
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "datetime":  pd.to_datetime(h["time"], utc=True),
        f"temp_f_{name}":     h["temperature_2m"],
        f"wind_mph_{name}":   h["windspeed_10m"],
        f"solar_rad_{name}":  h["shortwave_radiation"],
        f"precip_{name}":     h["precipitation"],
    })
    print(f"  ✓ Weather fetched for {name}: {len(df):,} rows")
    return df


def fetch_all_weather(start: str = START_DATE, end: str = END_DATE,
                      cache_path: Path = None) -> pd.DataFrame:
    """Fetch weather for all 4 CA locations and merge on datetime."""
    if cache_path and cache_path.exists():
        print(f"  → Loading cached weather from {cache_path.name}")
        return pd.read_parquet(cache_path)

    dfs = []
    for name, (lat, lon) in WEATHER_LOCATIONS.items():
        df = fetch_weather_location(name, lat, lon, start, end)
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="datetime", how="inner")

    # Compute CA-wide averages (population-weighted approximation)
    # Weights: LA (largest) > Sacramento > SF > Fresno
    W = {"los_angeles": 0.40, "sacramento": 0.25, "san_francisco": 0.20, "fresno": 0.15}
    merged["temp_f"] = sum(merged[f"temp_f_{n}"] * w for n, w in W.items())
    merged["wind_mph"] = sum(merged[f"wind_mph_{n}"] * w for n, w in W.items())
    merged["solar_rad"] = sum(merged[f"solar_rad_{n}"] * w for n, w in W.items())
    merged["temp_f_max"] = merged[[f"temp_f_{n}" for n in W]].max(axis=1)

    if cache_path:
        merged.to_parquet(cache_path, index=False)
        print(f"  → Weather cached to {cache_path.name}")
    return merged


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all ML features to the merged DataFrame.
    All features are deterministic functions of the raw data.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # ── Temporal features ──
    df["hour"]       = df["datetime"].dt.hour
    df["dow"]        = df["datetime"].dt.dayofweek      # 0=Mon, 6=Sun
    df["month"]      = df["datetime"].dt.month
    df["year"]       = df["datetime"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_summer"]  = df["month"].isin([6, 7, 8, 9]).astype(int)

    # ── Weather features ──
    df["temp_f_sq"]        = df["temp_f"] ** 2           # nonlinear AC load
    df["heat_stress"]      = (df["temp_f"] > 95).astype(int)
    df["extreme_heat"]     = (df["temp_f"] > 105).astype(int)
    df["temp_x_peak_hour"] = df["temp_f"] * df["hour"].isin(range(17, 22)).astype(int)

    # ── Lagged price features ──
    df["lmp_lag1h"]     = df["lmp"].shift(1)
    df["lmp_lag24h"]    = df["lmp"].shift(24)
    df["lmp_lag168h"]   = df["lmp"].shift(168)           # 1-week lag
    df["lmp_roll24_mean"] = df["lmp"].shift(1).rolling(24).mean()
    df["lmp_roll24_std"]  = df["lmp"].shift(1).rolling(24).std()
    df["lmp_roll168_mean"] = df["lmp"].shift(1).rolling(168).mean()
    df["lmp_zscore"]    = (
        (df["lmp"] - df["lmp_roll24_mean"]) /
        (df["lmp_roll24_std"] + 1e-9)
    )

    # ── Spike label (year-normalized top 5%) ──
    # Year-normalize to remove secular price trends (see Research Log Entry 004)
    df["spike"] = 0
    for year in df["year"].unique():
        mask = df["year"] == year
        threshold = df.loc[mask, "lmp"].quantile(0.95)
        df.loc[mask & (df["lmp"] >= threshold), "spike"] = 1
    print(f"  → Spike rate: {df['spike'].mean():.1%} (target ~5%)")

    # ── Heat event flag (3+ consecutive days with max temp > 95°F) ──
    # Used for subsetting in evaluation (not a training feature — would leak)
    daily_max = df.groupby(df["datetime"].dt.date)["temp_f"].max()
    hot_days = set(daily_max[daily_max > 95].index)

    def is_heat_event(dt):
        d = dt.date()
        # Check if this day and the two preceding days are all hot
        return all(
            (d - pd.Timedelta(days=i)).date() in hot_days
            for i in range(3)
        ) if hasattr(d, 'date') else False

    df["heat_event"] = df["datetime"].apply(
        lambda dt: int(
            all(
                (dt.date() - pd.Timedelta(days=i)) in hot_days
                for i in range(3)
            )
        )
    )
    print(f"  → Heat event hours: {df['heat_event'].sum():,} ({df['heat_event'].mean():.1%})")

    # ── Temperature regime bins (for stratified analysis) ──
    df["temp_regime"] = pd.cut(
        df["temp_f"],
        bins=[-np.inf, 75, 90, 100, np.inf],
        labels=["<75°F", "75-90°F", "90-100°F", ">100°F"]
    )

    return df


# ── Dataset Splits ────────────────────────────────────────────────────────────

def get_splits(df: pd.DataFrame) -> dict:
    """
    Return train / calibration / test splits.
    See Research Log Entry 004 for rationale.
    """
    train = df[df["year"].isin([2021, 2022])].copy()
    calib = df[df["year"] == 2023].copy()
    test  = df[df["year"] == 2024].copy()

    print(f"\n  Dataset splits:")
    print(f"  Train (2021-2022):    {len(train):>6,} rows | spike rate: {train['spike'].mean():.1%}")
    print(f"  Calibration (2023):  {len(calib):>6,} rows | spike rate: {calib['spike'].mean():.1%}")
    print(f"  Test (2024):         {len(test):>6,} rows  | spike rate: {test['spike'].mean():.1%}")
    return {"train": train, "calib": calib, "test": test}


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest(lmp_files: list, weather_cached: bool):
    """Write data provenance manifest for reproducibility."""
    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "caiso_node": "TH_NP15_GEN-APND",
        "caiso_market": "RTM (Real-Time Market)",
        "date_range": {"start": START_DATE, "end": END_DATE},
        "caiso_files": lmp_files,
        "weather_source": "Open-Meteo Archive API (https://open-meteo.com)",
        "weather_locations": {
            name: {"lat": lat, "lon": lon}
            for name, (lat, lon) in WEATHER_LOCATIONS.items()
        },
        "weather_cached": weather_cached,
        "spike_definition": "Top 5% of hourly LMP values within each calendar year",
        "train_years": [2021, 2022],
        "calibration_year": 2023,
        "test_year": 2024,
    }
    manifest_path = ROOT / "data" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  ✓ Manifest written to data/manifest.json")


# ── Master Load Function ──────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """Full pipeline: load CAISO + weather → merge → features → save."""
    print("=" * 60)
    print("Building research dataset")
    print("=" * 60)

    # 1. Load LMP prices
    print("\n[1/4] Loading CAISO LMP prices...")
    lmp = load_caiso_csvs()

    # 2. Fetch/load weather
    print("\n[2/4] Loading weather data...")
    weather_cache = PROCESSED_DIR / "weather_cache.parquet"
    weather = fetch_all_weather(cache_path=weather_cache)

    # 3. Merge
    print("\n[3/4] Merging datasets...")
    df = pd.merge(lmp, weather, on="datetime", how="inner")
    print(f"  → Merged rows: {len(df):,}")

    # 4. Feature engineering
    print("\n[4/4] Building features...")
    df = build_features(df)

    # 5. Save
    out_path = PROCESSED_DIR / "merged_hourly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")

    # 6. Write manifest
    caiso_files = [str(f) for f in RAW_DIR.glob("caiso_lmp_*.csv")]
    write_manifest(caiso_files, weather_cache.exists())

    print("\n" + "=" * 60)
    print(f"Dataset ready: {len(df):,} rows × {len(df.columns)} columns")
    print("=" * 60)
    return df


def load_processed() -> pd.DataFrame:
    """Load the pre-built processed dataset. Run build_dataset() first."""
    path = PROCESSED_DIR / "merged_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run: python data/load.py"
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_dataset()
    splits = get_splits(df)
    print("\nSample features:")
    feature_cols = ["datetime", "lmp", "spike", "temp_f", "wind_mph",
                    "lmp_zscore", "heat_stress", "heat_event", "temp_regime"]
    print(df[feature_cols].dropna().tail(5).to_string(index=False))
