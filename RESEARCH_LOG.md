# Research Log

**Project:** Marginal Coverage Can Mask Conditional Instability — Conformal Prediction for Electricity Spike Forecasting Under Weather and Community Heterogeneity
**Author:** Will Shin (UCSD HDSI) · **Period:** June–August 2026

Each entry records a decision, the alternatives considered, and why. This log is the source of truth behind the paper's methodology section.

---

## 001 · Topic framing

**Decision:** Frame as a conformal-prediction-under-distribution-shift study, not a plain forecasting benchmark.
**Alternatives:** weather→price correlation study (undergraduate-level); pure forecasting benchmark (crowded); RL dispatch optimization (out of scope for timeline).
**Why:** Conformal prediction gives a falsifiable question with a mathematical anchor (the exchangeability-based coverage guarantee) and produces a publishable result in either direction — coverage holds (robustness result) or degrades (failure-mode result).

## 002 · Data source pivot: CAISO LMP → EIA demand

**Decision:** Use EIA Open Data API v2 hourly CAISO demand (respondent `CISO`, type `D`), 2021–2024, instead of CAISO OASIS LMP prices.
**Why:** CAISO's free API rate-limits bulk historical pulls and does not reliably serve LMP data older than ~2 years; gridstatus retrievals yielded only ~22% coverage for 2023–24. EIA demand is complete (35,053 hourly rows, 0 missing), free, and includes the September 2022 heat wave. Demand spikes are also more directly tied to community impact (outage risk) than wholesale prices.
**Cost:** The target variable changed from price spikes to demand spikes; the conformal methodology is unchanged.

## 003 · Research question (formal)

Do split conformal prediction sets for California demand-spike forecasting maintain nominal coverage across temperature regimes — and do coverage failures concentrate in conditions associated with high-energy-burden communities?
H₀: coverage is uniform across regimes. H₁: coverage degrades in warm/hot regimes.

## 004 · Splits, spike label, and a construction note

**Splits (strictly chronological):** train 2021–22 (n=17,511) · calibration 2023 (n=8,758) · test 2024 (n=8,784).
**Spike label:** top 5% of demand *within each calendar year* (year-normalized to remove secular trends).
**Construction note (stated in paper as a Remark):** year-normalization matches the marginal spike rate (~5%) across splits by design, so conservative *marginal* coverage is partially guaranteed by construction. The scientifically meaningful result is therefore the *conditional* (per-regime) behavior.

## 005 · Feature set and leakage guard

**Decision:** Weather + calendar features only (12 features). All demand-, price-, and target-derived variables blocked from the classifier by a runtime guard.
**History:** An early run included `demand_lag1h` etc.; the classifier reached AUROC 0.997 with lag features carrying ~80% importance — autocorrelation, not weather signal. Lags were removed so the model answers the actual question (can weather predict spikes?).
**Verification:** shuffled-label negative control → test AUROC 0.562 (mild elevation above 0.5 attributed to temporal autocorrelation in weather covariates, not leakage).

## 006 · Conformal methods

- **Split conformal** (α=0.10): nonconformity s = 1 − p̂(spike) on 2023 calibration spikes (n=438) → global threshold τ = 0.0133.
- **Temperature-stratified (Mondrian)**: regime-specific thresholds for <75 / 75–90 / 90–100 / ≥100°F; the ≥100°F stratum has zero calibration spikes and falls back to global.
- **Conservative variant**: τᵣᶜᵒⁿˢ = min(τ_global, τᵣ) — prevents warm-regime stratification from inducing undercoverage.
- **ACI** (Gibbs & Candès 2021) with γ=0.005 as a sequential baseline.
- All coverage estimates reported with Wilson 95% CIs; cells with n<5 spikes marked insufficient support.

## 007 · Vulnerability layer and the geographic proxy

**Data:** DOE LEAD Tool v3 county-level energy burden, 25 CA counties (Tulare 7.65% … Marin 1.62%).
**Constraint:** demand is one system-wide number per hour — it cannot be geographically decomposed. So the audit classifies **hours, not places**.
**Proxy:** Fresno−SF temperature differential. Fresno (inland Central Valley, high burden) and SF (coastal, low burden) anchor opposite ends of the burden spectrum. Hours with differential ≥15°F are dominated by inland Central Valley heat — conditions under which high-burden communities drive demand. Applies to 32.5% of test hours.
**Framing rule:** all claims are about *conditions associated with* high-burden communities, never about specific counties. County LEAD data justifies the proxy's direction; it is not used to make county-level claims.

## 008 · Results and narrative pivot

Original hypothesis (coverage degrades in heat) was not confirmed in its simple form. Actual findings:
1. Marginal coverage is conservative: 96.8% at nominal 90%.
2. Raw stratification *reveals and induces* conditional instability: 86.2% at 75–90°F, Wilson CI [81.4, 89.9] entirely below nominal (thresholds 23–35× more aggressive than global).
3. Conservative stratification restores ~97.5% (analytical upper bound on the same test year).
4. ACI fails severely (35.5%) — γ misspecification under a conservative baseline (correction horizon 1/γ = 200 steps).
5. Covariate shift is mild but real (two-sample AUROC 0.577).
6. Spike support concentrates overwhelmingly in high-vulnerability-condition hours (103–246 per warm stratum vs ≤8 in low-vulnerability cells) — the support imbalance is itself the equity finding.

Paper reframed accordingly: **marginal validity can mask conditional instability**, and stratification is a diagnostic, not a free improvement.

## 009 · Venue

CCAI "Tackling Climate Change with ML" workshop @ NeurIPS 2026, Papers track (4 pages + references + appendix), double-blind, OpenReview. Full 12-page version to arXiv (cs.LG). Repo anonymized via anonymous.4open.science for review; real URL restored at camera-ready.
