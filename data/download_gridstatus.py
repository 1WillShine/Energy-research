import gridstatus
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

caiso = gridstatus.CAISO()

print("Downloading CAISO DAM LMP 2021-2024...")
print("(Day-Ahead market — more reliable historical coverage)\n")

lmp_df = caiso.get_lmp(
    start=pd.Timestamp("2021-01-01"),
    end=pd.Timestamp("2024-12-31"),
    market="DAY_AHEAD_HOURLY",
    locations=["TH_NP15_GEN-APND"],
    sleep=5,
)

print(f"\nRaw rows downloaded: {len(lmp_df):,}")
print(f"Columns: {list(lmp_df.columns)}")

# Normalize datetime and LMP column names
time_col  = next((c for c in lmp_df.columns if "time" in c.lower()), None)
price_col = next((c for c in lmp_df.columns if "lmp" in c.lower()), None)

print(f"Using time_col='{time_col}', price_col='{price_col}'")

lmp_df["datetime"] = pd.to_datetime(lmp_df[time_col], utc=True).dt.floor("h")
lmp_df["lmp"]      = pd.to_numeric(lmp_df[price_col], errors="coerce")
hourly = lmp_df.groupby("datetime")["lmp"].mean().reset_index()
hourly = hourly.dropna()

print(f"Hourly rows after aggregation: {len(hourly):,}")
print(f"Date range: {hourly.datetime.min()} to {hourly.datetime.max()}\n")

for year in range(2021, 2025):
    yr = hourly[hourly["datetime"].dt.year == year]
    out = RAW_DIR / f"caiso_lmp_{year}.csv"
    yr.to_csv(out, index=False)
    print(f"  {year}: {len(yr):,} rows → {out.name}")

print("\nDone. Next: cd .. && python data/load.py")

