"""
scripts/experiments.py — Core ML experiments for the paper.

Runs in order:
  1. Baseline classifier (GradientBoosting spike predictor)
  2. Split conformal prediction calibration
  3. Stratified conformal calibration (proposed method)
  4. Covariate shift detection
  5. Coverage evaluation across temperature regimes
  6. Granger causality tests

All results saved to results/experiment_log.json for paper figures.

Usage:
    python scripts/experiments.py
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── sklearn imports (no xgboost needed — GBM is equivalent for this) ──────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Feature columns for ML ────────────────────────────────────────────────────
FEATURE_COLS = [
    "hour", "dow", "month", "is_weekend", "is_summer",
    "temp_f", "temp_f_sq", "wind_mph", "solar_rad",
    "heat_stress", "extreme_heat", "temp_x_peak_hour",
    "lmp_lag1h", "lmp_lag24h", "lmp_lag168h",
    "lmp_roll24_mean", "lmp_roll24_std", "lmp_roll168_mean",
    "lmp_zscore",
]

SHIFT_FEATURE_COLS = [
    # Weather only — no price features (price shift is what we're trying to explain)
    "temp_f", "temp_f_sq", "wind_mph", "solar_rad",
    "heat_stress", "extreme_heat", "hour", "month", "is_summer",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def train_spike_classifier(train: pd.DataFrame, verbose: bool = True):
    """
    Train GradientBoosting spike classifier.
    Returns fitted model and feature importance dict.
    """
    X_train = train[FEATURE_COLS].dropna()
    y_train = train.loc[X_train.index, "spike"]

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    if verbose:
        print(f"  ✓ Classifier trained | n_estimators used: {model.n_estimators_}")
        print(f"  Top features: {list(importances.items())[:5]}")

    return model, importances


def evaluate_classifier(model, data: pd.DataFrame, label: str = "test") -> dict:
    """Standard classifier evaluation metrics."""
    X = data[FEATURE_COLS].dropna()
    y = data.loc[X.index, "spike"]
    probs = model.predict_proba(X)[:, 1]

    auroc = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)
    ap    = average_precision_score(y, probs)

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")

    results = {
        "split":    label,
        "n":        len(y),
        "spike_rate": float(y.mean()),
        "auroc":    round(auroc, 4),
        "brier":    round(brier, 4),
        "avg_precision": round(ap, 4),
        "calibration_fraction_pos": fraction_pos.tolist(),
        "calibration_mean_pred":    mean_pred.tolist(),
    }
    print(f"  [{label}] AUROC={auroc:.3f} | Brier={brier:.4f} | AP={ap:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def fit_conformal(model, calib: pd.DataFrame, alpha: float = 0.10) -> float:
    """
    Fit split conformal prediction for spike classification.

    Nonconformity score: s_i = 1 - p̂(x_i) for true spike examples.
    Quantile threshold q̂ gives coverage guarantee ≥ 1 - alpha.

    Returns q̂ (the conformal threshold on probability scores).
    """
    X_calib = calib[FEATURE_COLS].dropna()
    y_calib = calib.loc[X_calib.index, "spike"]

    # Nonconformity scores for true spikes only
    # (We're doing conformal classification: predict SET containing true class)
    probs = model.predict_proba(X_calib)[:, 1]

    # For the spike class: nonconformity = 1 - P(spike)
    # Lower nonconformity = model is more confident about this spike
    spike_probs = probs[y_calib == 1]
    nonconf_scores = 1 - spike_probs

    # Conformal quantile (finite-sample correction)
    n = len(nonconf_scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(nonconf_scores, level)

    # Threshold on probability: predict spike if p̂(x) >= 1 - q̂
    prob_threshold = 1 - q_hat

    print(f"  Conformal calibration (α={alpha}):")
    print(f"    n_calibration_spikes = {n}")
    print(f"    q̂ (nonconformity) = {q_hat:.4f}")
    print(f"    prob_threshold    = {prob_threshold:.4f}")

    return prob_threshold


def evaluate_conformal_coverage(model, data: pd.DataFrame,
                                 prob_threshold: float,
                                 label: str = "test",
                                 alpha: float = 0.10) -> dict:
    """
    Evaluate empirical coverage of conformal prediction set.
    Coverage = fraction of true spikes captured by prediction set.

    Also computes coverage stratified by temperature regime.
    """
    X = data[FEATURE_COLS].dropna()
    y = data.loc[X.index, "spike"]
    probs = model.predict_proba(X)[:, 1]
    temp_regime = data.loc[X.index, "temp_regime"]
    heat_event  = data.loc[X.index, "heat_event"]

    # Overall coverage
    true_spikes = y == 1
    predicted_spikes = probs >= prob_threshold
    coverage = predicted_spikes[true_spikes].mean()  # recall on spike class
    false_alarm = predicted_spikes[~true_spikes].mean()

    print(f"\n  [{label}] Conformal Coverage (nominal = {1-alpha:.0%}):")
    print(f"    Overall empirical coverage: {coverage:.3f} (target ≥ {1-alpha:.2f})")
    print(f"    False alarm rate: {false_alarm:.3f}")

    # Stratified coverage
    regimes = ["<75°F", "75-90°F", "90-100°F", ">100°F"]
    stratified = {}
    for regime in regimes:
        mask = (temp_regime == regime) & true_spikes
        if mask.sum() < 10:
            stratified[regime] = None
            continue
        cov = predicted_spikes[mask].mean()
        n   = mask.sum()
        stratified[regime] = {"coverage": round(float(cov), 4), "n_spikes": int(n)}
        print(f"    {regime:>10}: coverage={cov:.3f} (n={n})")

    # Heat event coverage
    heat_mask = (heat_event == 1) & true_spikes
    if heat_mask.sum() >= 5:
        heat_cov = predicted_spikes[heat_mask].mean()
        print(f"    Heat events: coverage={heat_cov:.3f} (n={heat_mask.sum()})")
    else:
        heat_cov = None

    return {
        "split": label,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": round(float(coverage), 4),
        "false_alarm_rate": round(float(false_alarm), 4),
        "prob_threshold": round(prob_threshold, 4),
        "stratified_coverage": stratified,
        "heat_event_coverage": round(float(heat_cov), 4) if heat_cov else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STRATIFIED CONFORMAL CALIBRATION (Proposed Method)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_stratified_conformal(model, calib: pd.DataFrame,
                              alpha: float = 0.10) -> dict:
    """
    Fit separate conformal thresholds per temperature regime.

    Returns dict mapping regime → prob_threshold.
    This is our proposed method vs. the single-threshold baseline.

    Theoretical note: provides conditional coverage within each stratum,
    not marginal coverage. Stated as limitation in paper.
    """
    regimes = ["<75°F", "75-90°F", "90-100°F", ">100°F"]
    thresholds = {}

    print(f"\n  Stratified conformal calibration (α={alpha}):")
    for regime in regimes:
        regime_data = calib[calib["temp_regime"] == regime]
        X = regime_data[FEATURE_COLS].dropna()
        y = regime_data.loc[X.index, "spike"]

        spike_mask = y == 1
        if spike_mask.sum() < 20:
            print(f"    {regime}: insufficient spikes ({spike_mask.sum()}), using global threshold")
            thresholds[regime] = None  # fall back to global
            continue

        probs = model.predict_proba(X)[:, 1]
        spike_probs = probs[spike_mask]
        nonconf = 1 - spike_probs

        n = len(nonconf)
        level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        q_hat = np.quantile(nonconf, level)
        prob_thresh = 1 - q_hat

        thresholds[regime] = prob_thresh
        print(f"    {regime:>10}: threshold={prob_thresh:.4f} (n_spikes={n})")

    return thresholds


def evaluate_stratified_coverage(model, data: pd.DataFrame,
                                  stratified_thresholds: dict,
                                  global_threshold: float,
                                  alpha: float = 0.10) -> dict:
    """Compare stratified vs. global conformal coverage per regime."""
    X = data[FEATURE_COLS].dropna()
    y = data.loc[X.index, "spike"]
    probs = model.predict_proba(X)[:, 1]
    temp_regime = data.loc[X.index, "temp_regime"]

    regimes = ["<75°F", "75-90°F", "90-100°F", ">100°F"]
    comparison = {}

    print(f"\n  Coverage comparison (stratified vs. global, nominal={1-alpha:.0%}):")
    print(f"  {'Regime':<12} {'Global':>8} {'Stratified':>12} {'Δ':>6}")
    print(f"  {'-'*42}")

    for regime in regimes:
        mask = (temp_regime == regime) & (y == 1)
        if mask.sum() < 5:
            continue

        # Global threshold coverage
        global_cov = (probs[mask] >= global_threshold).mean()

        # Stratified threshold coverage
        thresh = stratified_thresholds.get(regime, global_threshold)
        if thresh is None:
            thresh = global_threshold
        strat_cov = (probs[mask] >= thresh).mean()

        delta = strat_cov - global_cov
        comparison[regime] = {
            "n_spikes": int(mask.sum()),
            "global_coverage":     round(float(global_cov), 4),
            "stratified_coverage": round(float(strat_cov), 4),
            "delta": round(float(delta), 4),
        }
        print(f"  {regime:<12} {global_cov:>8.3f} {strat_cov:>12.3f} {delta:>+6.3f}")

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COVARIATE SHIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_covariate_shift(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Classifier two-sample test (Lopez-Paz & Oquendo 2017).
    Label train=0, test=1. Train classifier on weather features only.
    AUROC > 0.7 → strong detectable shift.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    train_X = train[SHIFT_FEATURE_COLS].dropna()
    test_X  = test[SHIFT_FEATURE_COLS].dropna()

    # Sample equally to avoid class imbalance
    n = min(len(train_X), len(test_X))
    train_sample = train_X.sample(n, random_state=42)
    test_sample  = test_X.sample(n, random_state=42)

    X_shift = pd.concat([train_sample, test_sample], ignore_index=True)
    y_shift = np.array([0] * n + [1] * n)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    cv_scores = cross_val_score(
        clf, X_shift, y_shift,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc"
    )

    auroc_mean = cv_scores.mean()
    auroc_std  = cv_scores.std()

    # Interpretation
    if auroc_mean > 0.70:
        interpretation = "Strong detectable shift"
    elif auroc_mean > 0.60:
        interpretation = "Moderate shift"
    elif auroc_mean > 0.55:
        interpretation = "Mild shift"
    else:
        interpretation = "No detectable shift"

    print(f"\n  Covariate shift detection (train=2021-22 vs test=2024):")
    print(f"    AUROC = {auroc_mean:.3f} ± {auroc_std:.3f}")
    print(f"    Interpretation: {interpretation}")

    return {
        "auroc_mean": round(float(auroc_mean), 4),
        "auroc_std":  round(float(auroc_std), 4),
        "interpretation": interpretation,
        "cv_scores": cv_scores.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRANGER CAUSALITY (simplified — statsmodels not available in all envs)
# ═══════════════════════════════════════════════════════════════════════════════

def granger_causality_manual(df: pd.DataFrame, max_lag: int = 24) -> dict:
    """
    Manual Granger causality test: does lagged temp help predict LMP changes
    beyond lagged LMP alone?

    Uses F-test comparing restricted model (AR of LMP) vs unrestricted
    model (AR of LMP + lagged temperature).

    Works without statsmodels.
    """
    from sklearn.linear_model import LinearRegression

    results = {}
    df_clean = df[["lmp", "temp_f"]].dropna().copy()
    df_clean["dlmp"] = df_clean["lmp"].diff()  # first-difference for stationarity

    for lag in [1, 6, 12, 24]:
        # Build lag matrix
        rows = []
        for i in range(lag, len(df_clean) - 1):
            row = {}
            for l in range(1, lag + 1):
                row[f"lmp_lag{l}"]  = df_clean["dlmp"].iloc[i - l]
                row[f"temp_lag{l}"] = df_clean["temp_f"].iloc[i - l]
            row["target"] = df_clean["dlmp"].iloc[i]
            rows.append(row)

        lag_df = pd.DataFrame(rows).dropna()
        y = lag_df["target"].values
        lmp_cols  = [c for c in lag_df if c.startswith("lmp_lag")]
        temp_cols = [c for c in lag_df if c.startswith("temp_lag")]

        # Restricted: AR(lag) of LMP only
        X_r = lag_df[lmp_cols].values
        reg_r = LinearRegression().fit(X_r, y)
        rss_r = np.sum((y - reg_r.predict(X_r)) ** 2)

        # Unrestricted: AR(lag) + temp lags
        X_u = lag_df[lmp_cols + temp_cols].values
        reg_u = LinearRegression().fit(X_u, y)
        rss_u = np.sum((y - reg_u.predict(X_u)) ** 2)

        # F-statistic
        n = len(y)
        k = lag  # number of temp lags added
        f_stat = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * lag - 1))

        # Approximate p-value from F distribution
        from scipy.stats import f as f_dist
        p_val = 1 - f_dist.cdf(f_stat, k, n - 2 * lag - 1)

        results[f"lag_{lag}h"] = {
            "f_statistic": round(float(f_stat), 3),
            "p_value":     round(float(p_val), 6),
            "significant": p_val < 0.05,
        }
        sig = "✓" if p_val < 0.05 else "✗"
        print(f"    Lag {lag:>2}h: F={f_stat:>7.2f}, p={p_val:.4f} {sig}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN — Run All Experiments
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(df: pd.DataFrame) -> dict:
    """Run complete experiment suite. Returns all results as nested dict."""
    from data.load import get_splits

    print("\n" + "="*60)
    print("Running experiment suite")
    print("="*60)

    splits = get_splits(df)
    train = splits["train"].dropna(subset=FEATURE_COLS + ["spike"])
    calib = splits["calib"].dropna(subset=FEATURE_COLS + ["spike"])
    test  = splits["test"].dropna(subset=FEATURE_COLS + ["spike"])

    results = {"run_at": datetime.utcnow().isoformat(), "experiments": {}}

    # ── Experiment 1: Baseline classifier ──
    print("\n[Exp 1] Training spike classifier...")
    model, importances = train_spike_classifier(train)
    train_metrics = evaluate_classifier(model, train, "train")
    test_metrics  = evaluate_classifier(model, test,  "test")
    results["experiments"]["classifier"] = {
        "train": train_metrics,
        "test":  test_metrics,
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
    }

    # ── Experiment 2: Standard conformal prediction ──
    print("\n[Exp 2] Fitting conformal prediction...")
    alpha = 0.10  # target 90% coverage
    prob_threshold = fit_conformal(model, calib, alpha=alpha)
    coverage_results = evaluate_conformal_coverage(
        model, test, prob_threshold, label="test", alpha=alpha
    )
    results["experiments"]["conformal_standard"] = {
        "alpha": alpha,
        "prob_threshold": prob_threshold,
        "coverage": coverage_results,
    }

    # ── Experiment 3: Stratified conformal ──
    print("\n[Exp 3] Stratified conformal calibration...")
    stratified_thresholds = fit_stratified_conformal(model, calib, alpha=alpha)
    comparison = evaluate_stratified_coverage(
        model, test, stratified_thresholds, prob_threshold, alpha=alpha
    )
    results["experiments"]["conformal_stratified"] = {
        "thresholds_per_regime": stratified_thresholds,
        "coverage_comparison": comparison,
    }

    # ── Experiment 4: Covariate shift ──
    print("\n[Exp 4] Covariate shift detection...")
    shift_results = detect_covariate_shift(train, test)
    results["experiments"]["covariate_shift"] = shift_results

    # ── Experiment 5: Granger causality ──
    print("\n[Exp 5] Granger causality tests (temp → LMP)...")
    granger_results = granger_causality_manual(train[["lmp", "temp_f"]])
    results["experiments"]["granger"] = granger_results

    # ── Experiment 6: Vulnerability × Coverage Analysis (KEY CONTRIBUTION) ──
    print("\n[Exp 6] Community vulnerability × conformal coverage analysis...")
    try:
        from data.vulnerability import (
            load_vulnerability, merge_vulnerability,
            analyze_vulnerability_coverage, vulnerability_descriptive_stats
        )
        vuln_df  = load_vulnerability()
        test_vuln, vuln_summary = merge_vulnerability(test.copy(), vuln_df)
        vuln_stats = vulnerability_descriptive_stats(vuln_df)

        vuln_coverage = analyze_vulnerability_coverage(
            model=model,
            test_df=test_vuln,
            feature_cols=FEATURE_COLS,
            global_threshold=prob_threshold,
            stratified_thresholds=stratified_thresholds,
            alpha=alpha,
        )
        results["experiments"]["vulnerability"] = {
            "descriptive_stats":  vuln_stats,
            "zone_summary":       vuln_summary,
            "coverage_matrix":    vuln_coverage,
        }
        print("  ✓ Vulnerability experiment complete")
    except Exception as e:
        print(f"  ⚠ Vulnerability experiment failed: {e}")
        results["experiments"]["vulnerability"] = {"error": str(e)}

    # ── Save results ──
    out_path = RESULTS_DIR / "experiment_log.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ All results saved to results/experiment_log.json")

    return results, model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.load import load_processed

    try:
        df = load_processed()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results, model = run_all_experiments(df)
    print("\n✓ Experiment suite complete.")
