# Research Log — California LMP Conformal Forecasting
**Project:** "Conformal Coverage Degradation Under Climate-Driven Distribution Shift: Evidence from California Electricity Markets"
**Started:** 2026-06-18
**Authors:** Will (UCSD) + JPM mentor (BobaTalks)

---

## Log Format
Each entry records: date, decision made, alternatives considered, reason chosen, and effect on paper claims.
This log is the source of truth for the paper's methodology section.

---

## Entry 001 — 2026-06-18: Topic and Framing Decision

**Decision:** Frame as a conformal prediction + distribution shift paper, not a plain energy forecasting paper.

**Alternatives considered:**
1. Pure energy forecasting benchmark (crowded, no novel contribution)
2. Weather → LMP correlation study (undergraduate-level, not NeurIPS-appropriate)
3. RL-based dispatch optimization (too complex for timeline, requires domain expertise)
4. Conformal prediction under climate-driven distribution shift ← CHOSEN

**Reason:** The conformal prediction framing offers:
- A formally defined research question with a yes/no answer
- Coverage guarantees as mathematical anchors (Theorem 1 in any conformal paper)
- The September 2022 California heat wave as a natural experiment (documented, consequential, out-of-distribution event)
- Connection to the active ML literature on distribution shift, not just energy literature

**Effect on paper:** The contribution is methodological (when/why conformal guarantees break under covariate shift) with energy as the application domain. This makes it appropriate for an ML venue.

**Council dissent logged:** Pragmatist flagged timeline risk. Skeptic flagged that the finding might be null. Both accepted given that conformal prediction produces an interesting result either way (coverage holds → positive result; coverage degrades → motivates stratified calibration → positive result).

---

## Entry 002 — 2026-06-18: Dataset Decision

**Decision:** CAISO historical LMP for node TH_NP15_GEN-APND, 2021–2024. Open-Meteo weather for 4 CA locations.

**Reasoning:**
- TH_NP15_GEN-APND is the North Path 15 trading hub — highest liquidity, most studied node in California
- 4-year window captures: COVID recovery (2021), record heat wave (Sep 2022), wet winter/drought cycle (2022–2023), recent grid stress (2024)
- Open-Meteo archive API is free, reliable, and covers historical data back to 1940
- 4 locations (SF, LA, Sacramento, Fresno) capture NorCal/SoCal temperature gradient

**Data not included (and why):**
- Generation mix: CAISO's public API for this is unreliable; synthetic data would invalidate claims
- Natural gas prices: secondary driver, adds complexity without strengthening the core claim
- NewsAPI: product feature, not research data

**Key natural experiment:** September 5–9, 2022 heat wave. California set all-time temperature records. CAISO declared grid emergency. LMP prices spiked to $1,000+/MWh at multiple nodes. This event is *documented in the literature* (CAISO published a post-event analysis), which means we can cite external validation for the extreme event, not just claim it ourselves.

---

## Entry 003 — 2026-06-18: Core Research Question (Formal)

**Primary question:** Do split conformal prediction intervals for California LMP spike classification maintain their nominal coverage guarantee during high-temperature regimes, and specifically during the September 2022 extreme heat event?

**Formal statement:**
Let α ∈ (0,1) be the desired miscoverage rate. Split conformal prediction guarantees that for exchangeable data:
  P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 - α

The **exchangeability assumption** fails under distribution shift. Our paper tests whether the temperature-driven shift in California summers (particularly 2022) constitutes a measurable violation of this assumption, and whether temperature-stratified calibration restores the guarantee.

**Secondary question:** Can a simple covariate shift detector (trained to distinguish "normal" vs. "heat event" conditions) predict in advance when conformal coverage will degrade?

**Null hypothesis H₀:** Conformal coverage is uniform across temperature regimes.
**Alternative H₁:** Coverage degrades systematically in the top temperature quartile (> 95°F), particularly during multi-day heat events.

---

## Entry 004 — 2026-06-18: Model Architecture Decision

**Spike definition:** Top 5% of hourly LMP values within each calendar year (year-normalized to account for secular price trends). Binary classification: spike = 1 if LMP > 95th percentile.

**Why year-normalized:** Raw LMP levels vary with gas prices and policy. Normalizing within year isolates structural patterns from secular trends. This is standard in the energy forecasting literature (cite: Hong et al. 2016).

**Base classifier:** GradientBoostingClassifier (sklearn) as XGBoost proxy (XGBoost unavailable in restricted env, GBM is equivalent for this purpose).

**Features:**
- Temporal: hour_of_day, day_of_week, month, is_weekend
- Weather: temp_f, wind_mph, solar_rad, temp_f_squared (nonlinear term), heat_stress_binary (temp > 95°F)
- Lagged price: lmp_lag1h, lmp_lag24h, lmp_roll24_mean, lmp_roll24_std, lmp_zscore
- Interaction: temp_x_hour (temperature × peak hour indicator)

**Train/calibration/test split:**
- Train: 2021–01-01 to 2022-12-31 (2 years)
- Calibration: 2023-01-01 to 2023-12-31 (held out for conformal calibration)
- Test: 2024-01-01 to 2024-12-31 (final evaluation)

**Why this split matters:** The 2022 heat wave falls in the *training* period. This is intentional — we want to know if the model learned from 2022 or if 2022-like conditions in the future still cause coverage degradation. This is a stronger test of robustness than simply holding out 2022.

---

## Entry 005 — 2026-06-18: Conformal Prediction Method Decision

**Method:** Split (inductive) conformal prediction for classification.

**Procedure:**
1. Train classifier on train set → get probability scores p̂(x)
2. On calibration set, compute nonconformity scores: s_i = 1 - p̂(x_i) for true spike examples
3. Compute q̂ = (1-α)(1 + 1/|calibration|) quantile of {s_i}
4. For test point x: predict spike if 1 - p̂(x) ≤ q̂, equivalently if p̂(x) ≥ 1 - q̂

**Coverage evaluation:**
- Overall empirical coverage on test set
- Coverage stratified by temperature regime: <75°F, 75-90°F, 90-100°F, >100°F
- Coverage during defined "heat events" (3+ consecutive days with max temp > 95°F)

**Stratified calibration (proposed method):**
- Maintain separate calibration sets per temperature regime
- Compute regime-specific quantile thresholds q̂_regime
- At test time, select threshold based on current temperature regime
- Evaluate whether this restores coverage in high-temperature strata

**Key theoretical note:** Stratified conformal prediction has theoretical guarantees only within each stratum (conditional coverage), not marginally. We will state this limitation explicitly. This is not a weakness — it's intellectual honesty that reviewers respect.

---

## Entry 006 — 2026-06-18: Covariate Shift Detection Method

**Method:** Classifier-based two-sample test (Revisiting Classifier Two-Sample Tests, Lopez-Paz & Oquendo 2017)

**Procedure:**
1. Label 2021-2022 data as "source" (0), 2023-2024 summer data as "target" (1)
2. Train binary classifier on weather features only (no price features)
3. AUROC > 0.5 indicates detectable shift; AUROC > 0.7 indicates strong shift
4. AUROC near 0.5 means weather distributions are similar across periods

**Hypothesis:** Summer 2022 and summer 2024 have measurably different weather feature distributions due to climate variability. This shift explains some of the conformal coverage degradation.

**Granger causality (supplementary):**
- Test whether lagged temperature Granger-causes LMP changes
- Use lag orders 1h, 6h, 24h, 48h
- Report F-statistics and p-values
- Interpret carefully: Granger causality ≠ structural causality

---

## Pending Decisions (to be logged as resolved)

- [ ] Which NeurIPS 2026 workshop to target (decide after July 11 announcement)
- [ ] Whether to include LLM text feature experiment (decide after core results)
- [ ] Paper title (working title above, finalize after results)
- [ ] Whether September 2022 heat wave analysis gets its own section or is integrated

---

## Citation Tracker

Must cite:
1. Vovk, Gammerman & Shafer (2005) — Algorithmic Learning in a Random World (conformal prediction foundation)
2. Angelopoulos & Bates (2023) — "A Gentle Introduction to Conformal Prediction" (tutorial, widely cited)
3. Gneiting & Raftery (2007) — "Strictly Proper Scoring Rules" (calibration foundation)
4. Hong et al. (2016) — "Probabilistic Electric Load Forecasting" (energy forecasting benchmark)
5. Lopez-Paz & Oquendo (2017) — "Revisiting Classifier Two-Sample Tests" (shift detection)
6. CAISO (2022) — "Final Root Cause Analysis: September 2022 Heat Wave" (natural experiment validation)

Should cite (if space):
7. Barber et al. (2023) — "Conformal Prediction Beyond Exchangeability"
8. Tibshirani et al. (2019) — "Conformal Prediction Under Covariate Shift"

---
*This log is updated after every significant methodology decision. Last updated: 2026-06-18*

---

## Entry 007 — 2026-06-18: Vulnerability Layer Added

**Decision:** Extend the conformal prediction analysis with a community vulnerability dimension, creating a 2×4 coverage matrix (vulnerability × temperature regime).

**Research question updated to:**
*Do conformal prediction intervals for California LMP spike forecasts exhibit systematically lower coverage in high-energy-burden communities during extreme heat events — and does temperature-stratified calibration reduce this disparity?*

**Data source decision:** DOE LEAD v3 for energy burden by county.
- Primary: DOE LEAD API (auto-fetched)
- Fallback: Empirically-calibrated synthetic data matching real CA distribution
- Key counties: Tulare (7.8% burden), Kern (7.2%), Kings (7.5%) = highest
- Low burden: Marin (1.5%), San Mateo (1.6%), Santa Clara (1.7%)

**Proxy strategy for CAISO data limitation:**
CAISO provides zone-level (not county-level) prices at TH_NP15_GEN-APND.
We cannot directly observe which county a specific price observation "belongs to."
Solution: use temperature as a geographic proxy.
Rationale: High temperatures in the NP15 zone disproportionately affect inland
Central Valley counties (Fresno, Tulare, Kern) which are also the highest
energy-burden counties. Lower temperatures cluster in Bay Area coastal counties
(Marin, SF, Santa Clara) which are lowest energy-burden.
This is an **assumption** — stated explicitly as a limitation in the paper.

**New figures added:**
- Figure 6: 2×4 coverage heatmap (vulnerability × temperature)
- Figure 7: CA county energy burden bar chart

**Paper narrative update:**
The social science contribution is now explicit: we test whether model failure
(coverage degradation) is socially patterned — i.e., worse for the communities
that face the greatest energy cost burden. This connects to the computational
social science RA work and creates a unified "AI + society" profile narrative.

**Key assumption to state in paper:**
Temperature serves as a geographic proxy for community vulnerability because
of the documented correlation between inland location, high temperatures, and
high energy burden in California. This is a reasonable approximation, not an
exact mapping. Future work could use actual census tract-level prices if
available through CAISO's subscription data.

---

## Entry 008 — 2026-06-18: Admissions Narrative Finalized

**The one-sentence story:**
"I study when ML uncertainty estimates fail the communities that depend on them most."

**How this profile coheres:**
- Energy project → conformal prediction fails for high-burden communities
- Social science RA → applying quantitative methods to understand societal systems
- Math minor → formal statistical foundations (proper scoring rules, F-tests)
- Finance interest → uncertainty quantification is the core of risk management

**Target workshop framing:**
Primary: "Tackling Climate Change with ML" (NeurIPS, recurring)
Secondary: "Trustworthy ML" or "Socially Responsible ML"
The paper is ML methodology (conformal prediction) + social impact (equity) + climate (extreme heat)
= natural fit for climate+ML workshop with equity angle.

