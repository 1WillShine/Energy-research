# Getting Started — Run This Today
## Energy-Intelligence-Agent → Research Paper Pipeline

This file tells you exactly what to do, in order, right now.

---

## STEP 0 — Copy These Files Into Your Repo (10 minutes)

```
your_repo/
├── data/
│   ├── load.py          ← copy from caiso_research/data/load.py
│   ├── raw/             ← create empty (CAISO CSVs go here)
│   └── processed/       ← create empty (auto-populated)
├── scripts/
│   ├── experiments.py   ← copy from caiso_research/scripts/experiments.py
│   └── figures.py       ← copy from caiso_research/scripts/figures.py
├── paper/
│   ├── paper.tex        ← copy from caiso_research/paper/paper.tex
│   └── references.bib   ← copy from caiso_research/paper/references.bib
├── results/             ← create empty (auto-populated)
├── figures/             ← create empty (auto-populated)
├── RESEARCH_LOG.md      ← copy from caiso_research/RESEARCH_LOG.md
└── requirements.txt     ← update (see below)
```

Update requirements.txt:
```
streamlit>=1.32.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
anthropic>=0.25.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
pyarrow>=14.0.0
```

---

## STEP 1 — Register EIA API Key (10 minutes)

1. Go to: https://www.eia.gov/opendata/register.php
2. Register (free, instant)
3. Get your API key
4. Save it somewhere safe — you'll add it to data/load.py when you add gas prices later

Not needed for Phase 1. Just do it now while it's on your mind.

---

## STEP 2 — Download CAISO LMP Data (1-2 hours)

This is the most important step. Real data unlocks everything.

### Option A: CAISO OASIS Web Portal (Recommended)
1. Go to: http://oasis.caiso.com/mrioasis/logon.do
2. Create free account
3. Query: 
   - Report: PRC_LMP
   - Market: RTM (Real-Time Market)  
   - Node: TH_NP15_GEN-APND
   - Date range: 2021-01-01 to 2021-12-31 (do one year at a time)
   - Format: CSV
4. Repeat for 2022, 2023, 2024
5. Save as: data/raw/caiso_lmp_2021.csv, caiso_lmp_2022.csv, etc.

### Option B: Kaggle (Faster but verify data quality)
Search Kaggle for "CAISO LMP" or "California electricity prices"
A dataset covering 2019-2023 exists — check it includes TH_NP15_GEN-APND
Verify columns match: datetime, price (in $/MWh)

### What the CSV should look like:
```
INTERVALSTARTTIME_GMT,NODE,MW,OPR_DT,...
2021-01-01T00:00:00-0000,TH_NP15_GEN-APND,42.13,...
2021-01-01T00:05:00-0000,TH_NP15_GEN-APND,41.87,...
```
(5-minute intervals — load.py handles aggregation to hourly)

---

## STEP 3 — Build the Dataset (5 minutes of your time, ~10 min to run)

Once you have the CSVs:
```bash
cd your_repo
python data/load.py
```

This will:
- Load CAISO CSVs from data/raw/
- Fetch Open-Meteo weather automatically (no key needed)
- Merge and compute all features
- Save to data/processed/merged_hourly.parquet
- Write data/manifest.json

Expected output:
```
[1/4] Loading CAISO LMP prices...
  ✓ Loaded caiso_lmp_2021.csv: 8,760 hourly rows
  ✓ Loaded caiso_lmp_2022.csv: 8,760 hourly rows
  ...
[2/4] Loading weather data...
  ✓ Weather fetched for san_francisco: 35,064 rows
  ...
[3/4] Merging datasets...
  → Merged rows: ~34,000
[4/4] Building features...
  → Spike rate: 5.0% (target ~5%)
  → Heat event hours: ~1,200 (3.5%)
```

---

## STEP 4 — Run Experiments (10 minutes to run)

```bash
python scripts/experiments.py
```

This runs all 5 experiments and saves results to results/experiment_log.json.

After this, you will have REAL numbers for every [FILL] in the paper.

---

## STEP 5 — Generate Figures (2 minutes)

```bash
python scripts/figures.py
```

Saves all figures to figures/. Open each PDF and look at them.

The most important figure: fig1_coverage_degradation
- If coverage drops in the ">100°F" bar relative to "<75°F" → your hypothesis confirmed
- If coverage is roughly flat → you have a null result (still interesting! explains why)

---

## STEP 6 — Fill In the Paper

Open paper/paper.tex.
Every [FILL] and [X.X%] placeholder gets replaced with real numbers from:
- results/experiment_log.json (for quantitative results)
- Your own analysis of what the figures show (for interpretation)

The paper structure is complete. You are writing results and discussion,
not figuring out structure.

---

## STEP 7 — Troubleshooting Common Issues

### CAISO API returns empty data
Use the web portal download instead of the API.
The API has rate limits and requires specific datetime formatting.
Web portal downloads are more reliable.

### Weather fetch fails
Open-Meteo sometimes has API issues. Try again after 30 minutes.
Or use a VPN if you're getting rate-limited.

### Spike rate is not ~5%
Check that year-normalization is working.
Print: df.groupby('year')['spike'].mean() — should be ~0.05 each year.
If spike rate is very different, check that lmp values are in $/MWh not ¢/MWh.

### Coverage results are near 90% everywhere (null result)
This is also publishable. It means conformal prediction is robust to
this type of shift. Write it up honestly.
The paper claim becomes: "contrary to our hypothesis, conformal coverage
is maintained under temperature-driven shift in California electricity
markets, suggesting [mechanism]."

### Too few data points in ">100°F" regime
California rarely hits 100°F+ at the population-weighted composite level.
Use max temperature instead of average: change temp_regime to use temp_f_max.
This is logged in RESEARCH_LOG.md as a potential methodology change.

---

## Timeline Reminder

Phase A (now → Jun 28): Steps 1-5 complete. Real data flowing, experiments run.
Phase B (Jun 29 → Jul 25): Fill in paper, iterate with mentor, CSE 151A.
Phase C (Jul 25 → Aug 29): Polish paper, clean repo, submit to NeurIPS workshop.

---

## When You Have Results — What to Do Next

1. Open RESEARCH_LOG.md and add Entry 007 with your actual findings
2. If coverage degrades as hypothesized → proceed with paper as written
3. If coverage doesn't degrade → update paper framing (null result is honest)
4. Email your professor/CSE 151A instructor: "I'm running experiments on X,
   preliminary results show Y, would you be willing to look at a draft?"
5. Share draft with JPM mentor around Jul 12

---

*Built in collaboration session 2026-06-18. See RESEARCH_LOG.md for all methodology decisions.*
