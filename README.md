# Marginal Coverage Can Mask Conditional Instability

**Conformal Prediction for Electricity Spike Forecasting Under Weather and Community Heterogeneity**

*NeurIPS Workshop Submission, 2026*

---

## Key Finding

Standard split conformal prediction achieves 96.8% marginal spike coverage at a nominal 90% target — but this masks conditional instability. Temperature-stratified calibration reveals coverage of 86.2% in the 75–90°F regime (Wilson 95% CI [81.4%, 89.9%]), entirely below nominal. Spike events concentrate disproportionately in high-energy-burden inland California communities, making reliable coverage in warm regimes an environmental-justice concern.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download EIA demand data (requires internet, ~5 min)
python data/download_eia_demand.py

# 3. Build merged dataset with weather
python data/load_demand.py

# 4. Run all experiments → saves results/experiment_log.json
python scripts/experiments.py

# 5. Generate all paper figures → saves to figures/
python scripts/figures.py
```

## Repository Structure
├── data/
│   ├── download_eia_demand.py   # EIA Open Data API v2 downloader
│   ├── load_demand.py           # Merge demand + weather, build features
│   ├── vulnerability.py         # DOE LEAD energy burden data
│   └── raw/                     # Downloaded CSVs (gitignored)
├── scripts/
│   ├── experiments.py           # All 6 experiments → experiment_log.json
│   └── figures.py               # All paper figures from experiment results
├── paper/
│   └── paper.tex                # NeurIPS workshop submission (LaTeX)
├── results/
│   └── experiment_log.json      # Full experiment outputs (auto-generated)
├── figures/                     # All PDF/PNG figures (auto-generated)
└── RESEARCH_LOG.md              # Methodology decisions and rationale
## Data Sources

- **Grid demand**: EIA Open Data API v2, respondent CISO (California ISO), 2021–2024
- **Weather**: Open-Meteo historical archive API, 4 CA locations (SF, LA, Sacramento, Fresno)
- **Community vulnerability**: DOE LEAD Tool v3, county-level energy burden

All data is freely available. The download scripts handle everything automatically.

## Reproducibility

All experiments use Python 3 with scikit-learn and random seed 42. Results are deterministic given the same data inputs. See `RESEARCH_LOG.md` for all methodology decisions.

## Citation

```bibtex
@article{shin2026marginal,
  title={Marginal Coverage Can Mask Conditional Instability: Conformal Prediction for Electricity Spike Forecasting Under Weather and Community Heterogeneity},
  author={Shin, Will},
  journal={NeurIPS Workshop},
  year={2026},
  url={https://github.com/1WillShine/Energy-Research}
}
```
