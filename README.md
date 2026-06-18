# Conformal Coverage Under Climate-Driven Distribution Shift
## Evidence from California Electricity Markets

**Research question:** Do conformal prediction intervals for California LMP spike forecasts exhibit systematically lower coverage in high-energy-burden communities during extreme heat events?

### Setup
```bash
pip install -r requirements.txt
python data/load.py        # build dataset (requires CAISO CSVs in data/raw/)
python scripts/experiments.py  # run all experiments
python scripts/figures.py      # generate paper figures
```

### Data
- CAISO LMP: TH_NP15_GEN-APND, 2021–2024 (download from oasis.caiso.com)
- Weather: Open-Meteo archive API (auto-fetched)
- Community vulnerability: DOE LEAD v3 (auto-fetched)
