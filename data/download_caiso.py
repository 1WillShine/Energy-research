"""
data/download_caiso.py — Bulk download CAISO LMP data for TH_NP15_GEN-APND.

Handles the 31-day API limit automatically by downloading month by month.
Saves one CSV per month to data/raw/caiso_lmp_YYYY_MM.csv

Usage:
    python data/download_caiso.py

Takes ~20-40 minutes for 4 years. Let it run in background.
Progress is saved — if interrupted, re-run and it skips already-downloaded files.
"""

import requests
import pandas as pd
import zipfile
import io
import time
from pathlib import Path
from datetime import date, timedelta
import calendar

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CAISO_NODE = "TH_NP15_GEN-APND"
START_YEAR, START_MONTH = 2021, 1
END_YEAR,   END_MONTH   = 2024, 12


def download_month(year: int, month: int) -> pd.DataFrame | None:
    """Download one month of 5-min RTM LMP data from CAISO OASIS."""
    out_path = RAW_DIR / f"caiso_lmp_{year}_{month:02d}.csv"

    if out_path.exists():
        print(f"  ✓ {year}-{month:02d} already downloaded, skipping")
        return pd.read_csv(out_path)

    last_day = calendar.monthrange(year, month)[1]
    start_dt = f"{year}{month:02d}01T00:00-0000"
    end_dt   = f"{year}{month:02d}{last_day:02d}T23:59-0000"

    params = {
        "queryname":      "PRC_LMP",
        "startdatetime":  start_dt,
        "enddatetime":    end_dt,
        "version":        1,
        "market_run_id":  "RTM",
        "node":           CAISO_NODE,
        "resultformat":   6,       # CSV inside ZIP
    }

    try:
        resp = requests.get(
            "http://oasis.caiso.com/oasisapi/SingleZip",
            params=params,
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"  ✗ {year}-{month:02d}: HTTP {resp.status_code}")
            return None

        if len(resp.content) < 200:
            print(f"  ✗ {year}-{month:02d}: Empty response (CAISO may be throttling)")
            return None

        # Unzip and read CSV
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False)

        # Keep only what we need
        time_col  = next((c for c in df.columns if "INTERVALSTART" in c.upper()), None)
        price_col = next((c for c in df.columns
                         if c.upper() in ("MW", "VALUE") or "LMP_PRC" in c.upper()), None)

        if not time_col or not price_col:
            print(f"  ✗ {year}-{month:02d}: Unexpected columns: {list(df.columns)[:8]}")
            return None

        df = df[[time_col, price_col]].copy()
        df.columns = ["datetime", "lmp"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["lmp"]      = pd.to_numeric(df["lmp"], errors="coerce")
        df = df.dropna()

        # Aggregate 5-min → hourly
        df["datetime"] = df["datetime"].dt.floor("h")
        df = df.groupby("datetime")["lmp"].mean().reset_index()

        df.to_csv(out_path, index=False)
        print(f"  ✓ {year}-{month:02d}: {len(df):,} hourly rows → {out_path.name}")
        return df

    except zipfile.BadZipFile:
        print(f"  ✗ {year}-{month:02d}: Bad ZIP (CAISO returned HTML error page)")
        return None
    except Exception as e:
        print(f"  ✗ {year}-{month:02d}: {type(e).__name__}: {e}")
        return None


def consolidate_to_annual():
    """Merge monthly CSVs into annual files that load.py expects."""
    print("\nConsolidating monthly → annual files...")

    for year in range(START_YEAR, END_YEAR + 1):
        monthly_files = sorted(RAW_DIR.glob(f"caiso_lmp_{year}_*.csv"))
        if not monthly_files:
            print(f"  ⚠ No data for {year}")
            continue

        frames = [pd.read_csv(f) for f in monthly_files]
        annual = pd.concat(frames, ignore_index=True)
        annual["datetime"] = pd.to_datetime(annual["datetime"], utc=True)
        annual = annual.sort_values("datetime").drop_duplicates("datetime")

        out = RAW_DIR / f"caiso_lmp_{year}.csv"
        annual.to_csv(out, index=False)
        print(f"  ✓ {year}: {len(annual):,} hourly rows → {out.name}")


def main():
    print("=" * 55)
    print(f"CAISO LMP Bulk Downloader")
    print(f"Node: {CAISO_NODE}")
    print(f"Range: {START_YEAR}-01 to {END_YEAR}-12")
    print(f"Output: {RAW_DIR}/")
    print("=" * 55)
    print("(Takes 20-40 min total. Safe to interrupt and re-run.)\n")

    total   = (END_YEAR - START_YEAR) * 12 + (END_MONTH - START_MONTH + 1)
    success = 0
    failed  = []

    year, month = START_YEAR, START_MONTH
    for _ in range(total):
        df = download_month(year, month)
        if df is not None:
            success += 1
        else:
            failed.append(f"{year}-{month:02d}")

        # Advance month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1

        # Polite delay — CAISO rate-limits aggressive scrapers
        time.sleep(1.5)

    print(f"\n{'='*55}")
    print(f"Done: {success}/{total} months downloaded")
    if failed:
        print(f"Failed months: {failed}")
        print("Re-run the script to retry failed months.")

    if success > 0:
        consolidate_to_annual()
        print("\n✓ Annual files ready in data/raw/")
        print("  Next: python data/load.py")


if __name__ == "__main__":
    main()
