"""
scripts/diagnostics.py

Additional diagnostic experiments for the conformal electricity spike paper.

Runs five high-rigor audits:

1. Calendar-only AUROC
2. Weather-only AUROC without calendar features
3. Shuffled-label negative control
4. Coverage Wilson confidence intervals
5. Conservative stratified conformal variant

Outputs:
    results/diagnostic_log.json

Usage:
    python scripts/diagnostics.py
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


TEMP_REGIMES = ["<75°F", "75-90°F", "90-100°F", ">100°F"]


FULL_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_summer",
    "temp_f",
    "temp_f_sq",
    "wind_mph",
    "solar_rad",
    "heat_stress",
    "extreme_heat",
    "temp_x_peak_hour",
]

CALENDAR_ONLY_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_summer",
]

WEATHER_ONLY_NO_CALENDAR_FEATURES = [
    "temp_f",
    "temp_f_sq",
    "wind_mph",
    "solar_rad",
    "heat_stress",
    "extreme_heat",
]

TEMPERATURE_ONLY_FEATURES = [
    "temp_f",
    "temp_f_sq",
    "heat_stress",
    "extreme_heat",
]

WEATHER_NO_SOLAR_NO_CALENDAR_FEATURES = [
    "temp_f",
    "temp_f_sq",
    "wind_mph",
    "heat_stress",
    "extreme_heat",
]


LEAKAGE_COLS = {
    "demand_mw",
    "demand_zscore",
    "demand_roll24_mean",
    "demand_roll24_std",
    "demand_forecast_mw",
    "demand_error",
    "demand_lag1h",
    "demand_lag24h",
    "demand_lag168h",
    "lmp",
    "lmp_zscore",
    "lmp_roll24_mean",
    "lmp_roll24_std",
    "lmp_lag1h",
    "lmp_lag6h",
    "lmp_lag24h",
    "spike_lag1h",
    "spike_lag24h",
    "spike_roll24",
}


def ensure_temp_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure temp_regime exists.
    """
    df = df.copy()

    if "temp_regime" in df.columns:
        return df

    if "temp_f" not in df.columns:
        raise KeyError("Cannot create temp_regime because temp_f is missing.")

    bins = [-np.inf, 75, 90, 100, np.inf]
    labels = ["<75°F", "75-90°F", "90-100°F", ">100°F"]

    df["temp_regime"] = pd.cut(
        df["temp_f"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype(str)

    return df


def audit_features(feature_cols):
    """
    Fail if leakage columns are accidentally included.
    """
    bad = sorted(set(feature_cols).intersection(LEAKAGE_COLS))

    if bad:
        raise ValueError(f"Leakage columns included in feature set: {bad}")


def available_features(df: pd.DataFrame, feature_cols):
    """
    Keep only features present in dataframe.
    """
    return [c for c in feature_cols if c in df.columns]


def make_xy(df: pd.DataFrame, feature_cols, target_col="spike"):
    audit_features(feature_cols)

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    idx = X.dropna().index

    return X.loc[idx], y.loc[idx]


def make_model():
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )


def evaluate_probs(y_true, probs):
    out = {
        "n": int(len(y_true)),
        "spike_rate": float(np.mean(y_true)),
    }

    if len(np.unique(y_true)) < 2:
        out.update({
            "auroc": None,
            "avg_precision": None,
            "brier": None,
        })
        return out

    out.update({
        "auroc": round(float(roc_auc_score(y_true, probs)), 4),
        "avg_precision": round(float(average_precision_score(y_true, probs)), 4),
        "brier": round(float(brier_score_loss(y_true, probs)), 4),
    })

    return out


def train_and_evaluate_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols,
    label: str,
):
    """
    Train GBM on one feature set and evaluate train/test performance.
    """
    feature_cols = available_features(train, feature_cols)
    audit_features(feature_cols)

    X_train, y_train = make_xy(train, feature_cols)
    X_test, y_test = make_xy(test, feature_cols)

    model = make_model()
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return {
        "label": label,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "train": evaluate_probs(y_train, train_probs),
        "test": evaluate_probs(y_test, test_probs),
        "feature_importances": {
            k: round(float(v), 4) for k, v in importances.items()
        },
    }


def shuffled_label_negative_control(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols,
    n_repeats: int = 10,
):
    """
    Negative control.

    Shuffle y_train, train model, evaluate against real y_test.
    A valid pipeline should produce AUROC close to 0.50 on average.
    """
    feature_cols = available_features(train, feature_cols)
    audit_features(feature_cols)

    X_train, y_train = make_xy(train, feature_cols)
    X_test, y_test = make_xy(test, feature_cols)

    rows = []

    for seed in range(n_repeats):
        rng = np.random.default_rng(seed)
        y_shuffled = y_train.copy().values
        rng.shuffle(y_shuffled)

        model = make_model()
        model.set_params(random_state=seed)
        model.fit(X_train, y_shuffled)

        test_probs = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_probs(y_test, test_probs)
        metrics["seed"] = seed
        rows.append(metrics)

    aurocs = [r["auroc"] for r in rows if r["auroc"] is not None]
    aps = [r["avg_precision"] for r in rows if r["avg_precision"] is not None]

    return {
        "label": "shuffled_label_negative_control",
        "features": feature_cols,
        "n_repeats": n_repeats,
        "runs": rows,
        "summary": {
            "auroc_mean": round(float(np.mean(aurocs)), 4) if aurocs else None,
            "auroc_std": round(float(np.std(aurocs)), 4) if aurocs else None,
            "avg_precision_mean": round(float(np.mean(aps)), 4) if aps else None,
            "avg_precision_std": round(float(np.std(aps)), 4) if aps else None,
        },
        "interpretation": (
            "Expected AUROC should be near 0.50. If substantially above 0.55, "
            "there may be leakage or temporal/target construction artifacts."
        ),
    }


def wilson_ci(k: int, n: int, z: float = 1.96):
    """
    Wilson binomial confidence interval.
    """
    if n == 0:
        return None, None

    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom

    return float(center - half), float(center + half)


def coverage_summary_from_bool(covered, n_total):
    """
    Return coverage and Wilson CI.
    """
    n_total = int(n_total)

    if n_total == 0:
        return {
            "coverage": None,
            "n_spikes": 0,
            "n_covered": 0,
            "wilson_95_lo": None,
            "wilson_95_hi": None,
        }

    k = int(np.sum(covered))
    cov = k / n_total
    lo, hi = wilson_ci(k, n_total)

    return {
        "coverage": round(float(cov), 4),
        "n_spikes": n_total,
        "n_covered": k,
        "wilson_95_lo": round(float(lo), 4),
        "wilson_95_hi": round(float(hi), 4),
    }


def fit_conformal_threshold(model, calib: pd.DataFrame, feature_cols, alpha=0.10):
    X_calib, y_calib = make_xy(calib, feature_cols)

    probs = model.predict_proba(X_calib)[:, 1]
    spike_probs = probs[y_calib.values == 1]

    if len(spike_probs) == 0:
        return None

    nonconf = 1.0 - spike_probs

    n = len(nonconf)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)

    q_hat = np.quantile(nonconf, level)
    threshold = 1.0 - q_hat

    return float(threshold)


def fit_stratified_thresholds(
    model,
    calib: pd.DataFrame,
    feature_cols,
    alpha=0.10,
    min_calibration_spikes=20,
):
    calib = ensure_temp_regime(calib)
    thresholds = {}

    for regime in TEMP_REGIMES:
        sub = calib[calib["temp_regime"] == regime].copy()

        if len(sub) == 0:
            thresholds[regime] = None
            continue

        X_sub, y_sub = make_xy(sub, feature_cols)
        n_spikes = int((y_sub.values == 1).sum())

        if n_spikes < min_calibration_spikes:
            thresholds[regime] = None
            continue

        probs = model.predict_proba(X_sub)[:, 1]
        spike_probs = probs[y_sub.values == 1]

        nonconf = 1.0 - spike_probs
        n = len(nonconf)

        level = np.ceil((1 - alpha) * (n + 1)) / n
        level = min(level, 1.0)

        q_hat = np.quantile(nonconf, level)
        thresholds[regime] = float(1.0 - q_hat)

    return thresholds


def evaluate_threshold_policy(
    model,
    data: pd.DataFrame,
    feature_cols,
    global_threshold: float,
    thresholds_by_regime=None,
    label="policy",
):
    """
    Evaluate spike coverage under a threshold policy.

    If thresholds_by_regime is None, use global_threshold everywhere.
    Otherwise, use regime-specific threshold when available, else global.
    """
    data = ensure_temp_regime(data)

    X, y = make_xy(data, feature_cols)
    probs = model.predict_proba(X)[:, 1]

    regimes = data.loc[X.index, "temp_regime"].values

    if thresholds_by_regime is None:
        thresholds = np.array([global_threshold] * len(X), dtype=float)
    else:
        thresholds = np.array([
            thresholds_by_regime.get(regime)
            if thresholds_by_regime.get(regime) is not None
            else global_threshold
            for regime in regimes
        ], dtype=float)

    predicted_spike = probs >= thresholds
    true_spike = y.values == 1

    overall = coverage_summary_from_bool(
        predicted_spike[true_spike],
        int(true_spike.sum()),
    )

    false_alarm_rate = (
        float(predicted_spike[~true_spike].mean())
        if int((~true_spike).sum()) > 0
        else None
    )

    by_regime = {}

    for regime in TEMP_REGIMES:
        mask = (regimes == regime) & true_spike
        n = int(mask.sum())

        by_regime[regime] = coverage_summary_from_bool(
            predicted_spike[mask],
            n,
        )

        if thresholds_by_regime is None:
            threshold_used = global_threshold
            threshold_source = "global"
        else:
            raw = thresholds_by_regime.get(regime)
            if raw is None:
                threshold_used = global_threshold
                threshold_source = "global_fallback"
            else:
                threshold_used = raw
                threshold_source = "regime_specific"

        by_regime[regime]["threshold_used"] = (
            round(float(threshold_used), 4)
            if threshold_used is not None
            else None
        )
        by_regime[regime]["threshold_source"] = threshold_source

    return {
        "label": label,
        "overall": overall,
        "false_alarm_rate": (
            round(float(false_alarm_rate), 4)
            if false_alarm_rate is not None
            else None
        ),
        "by_regime": by_regime,
    }


def run_coverage_ci_and_conservative_stratified(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols,
    alpha=0.10,
):
    """
    Fit full model and compare:
      - standard global split conformal
      - raw stratified conformal
      - conservative stratified conformal

    Conservative stratified threshold:
        threshold_r = min(global_threshold, raw_threshold_r)

    Since a lower probability threshold means a larger prediction set,
    this is coverage-preserving relative to aggressive stratification.
    """
    feature_cols = available_features(train, feature_cols)
    audit_features(feature_cols)

    X_train, y_train = make_xy(train, feature_cols)

    model = make_model()
    model.fit(X_train, y_train)

    global_threshold = fit_conformal_threshold(
        model,
        calib,
        feature_cols,
        alpha=alpha,
    )

    raw_stratified = fit_stratified_thresholds(
        model,
        calib,
        feature_cols,
        alpha=alpha,
        min_calibration_spikes=20,
    )

    conservative_stratified = {}

    for regime, raw_threshold in raw_stratified.items():
        if raw_threshold is None:
            conservative_stratified[regime] = None
        else:
            conservative_stratified[regime] = min(
                float(global_threshold),
                float(raw_threshold),
            )

    standard_eval = evaluate_threshold_policy(
        model,
        test,
        feature_cols,
        global_threshold=global_threshold,
        thresholds_by_regime=None,
        label="standard_global",
    )

    raw_strat_eval = evaluate_threshold_policy(
        model,
        test,
        feature_cols,
        global_threshold=global_threshold,
        thresholds_by_regime=raw_stratified,
        label="raw_stratified",
    )

    conservative_eval = evaluate_threshold_policy(
        model,
        test,
        feature_cols,
        global_threshold=global_threshold,
        thresholds_by_regime=conservative_stratified,
        label="conservative_stratified",
    )

    return {
        "alpha": alpha,
        "nominal_coverage": round(float(1 - alpha), 4),
        "global_threshold": round(float(global_threshold), 4),
        "raw_stratified_thresholds": {
            k: round(float(v), 4) if v is not None else None
            for k, v in raw_stratified.items()
        },
        "conservative_stratified_thresholds": {
            k: round(float(v), 4) if v is not None else None
            for k, v in conservative_stratified.items()
        },
        "standard_global": standard_eval,
        "raw_stratified": raw_strat_eval,
        "conservative_stratified": conservative_eval,
        "interpretation": (
            "Raw stratification can improve efficiency but may under-cover in "
            "finite samples. Conservative stratification uses min(global, raw) "
            "thresholds, making prediction sets at least as inclusive as the "
            "global method within supported strata."
        ),
    }


def print_feature_ablation_table(ablation_results):
    print("\nFeature ablation diagnostics")
    print("-" * 78)
    print(f"{'Model':<36} {'n_feat':>6} {'Train AUROC':>12} {'Test AUROC':>11} {'Test AP':>9}")
    print("-" * 78)

    for row in ablation_results:
        print(
            f"{row['label']:<36} "
            f"{row['n_features']:>6} "
            f"{row['train']['auroc']:>12.4f} "
            f"{row['test']['auroc']:>11.4f} "
            f"{row['test']['avg_precision']:>9.4f}"
        )


def print_coverage_table(coverage_results):
    print("\nCoverage diagnostics with Wilson 95% intervals")
    print("-" * 100)

    for policy_name in [
        "standard_global",
        "raw_stratified",
        "conservative_stratified",
    ]:
        policy = coverage_results[policy_name]

        print(f"\n{policy_name}")
        print(
            f"Overall coverage = {policy['overall']['coverage']} "
            f"[{policy['overall']['wilson_95_lo']}, "
            f"{policy['overall']['wilson_95_hi']}], "
            f"n={policy['overall']['n_spikes']}, "
            f"false alarm={policy['false_alarm_rate']}"
        )

        print(f"{'Regime':<12} {'Coverage':>10} {'95% CI':>22} {'n':>6} {'Threshold':>10}")
        print("-" * 70)

        for regime in TEMP_REGIMES:
            cell = policy["by_regime"][regime]

            cov = cell["coverage"]
            lo = cell["wilson_95_lo"]
            hi = cell["wilson_95_hi"]
            n = cell["n_spikes"]
            th = cell["threshold_used"]

            if cov is None:
                cov_str = "N/A"
                ci_str = "N/A"
            else:
                cov_str = f"{cov:.4f}"
                ci_str = f"[{lo:.4f}, {hi:.4f}]"

            print(f"{regime:<12} {cov_str:>10} {ci_str:>22} {n:>6} {th:>10}")


def run_all_diagnostics():
    import sys
    sys.path.insert(0, str(ROOT))

    from data.load_demand import load_processed, get_splits

    print("\n" + "=" * 80)
    print("Running diagnostic experiments")
    print("=" * 80)

    df = load_processed()
    df = ensure_temp_regime(df)

    splits = get_splits(df)

    train = splits["train"].dropna(subset=["spike"]).copy()
    calib = splits["calib"].dropna(subset=["spike"]).copy()
    test = splits["test"].dropna(subset=["spike"]).copy()

    print("\nSplit sizes")
    print(f"  Train: {len(train):,}")
    print(f"  Calib: {len(calib):,}")
    print(f"  Test:  {len(test):,}")

    results = {
        "run_at": datetime.utcnow().isoformat(),
        "diagnostics": {},
    }

    # 1 and 2 plus extra ablations
    print("\n[1-2] Feature ablation diagnostics...")

    ablation_specs = [
        ("full_weather_calendar", FULL_FEATURES),
        ("calendar_only", CALENDAR_ONLY_FEATURES),
        ("weather_only_no_calendar", WEATHER_ONLY_NO_CALENDAR_FEATURES),
        ("temperature_only", TEMPERATURE_ONLY_FEATURES),
        ("weather_no_solar_no_calendar", WEATHER_NO_SOLAR_NO_CALENDAR_FEATURES),
    ]

    ablation_results = []

    for label, cols in ablation_specs:
        print(f"  Running: {label}")
        row = train_and_evaluate_feature_set(train, test, cols, label)
        ablation_results.append(row)

    results["diagnostics"]["feature_ablations"] = ablation_results
    print_feature_ablation_table(ablation_results)

    # 3. Shuffled label negative control
    print("\n[3] Shuffled-label negative control...")
    shuffled = shuffled_label_negative_control(
        train,
        test,
        FULL_FEATURES,
        n_repeats=10,
    )
    results["diagnostics"]["shuffled_label_negative_control"] = shuffled

    print(
        f"  Shuffled-label AUROC mean = "
        f"{shuffled['summary']['auroc_mean']} ± {shuffled['summary']['auroc_std']}"
    )
    print(
        f"  Shuffled-label AP mean = "
        f"{shuffled['summary']['avg_precision_mean']} ± "
        f"{shuffled['summary']['avg_precision_std']}"
    )

    # 4 and 5. Coverage CIs and conservative stratified conformal
    print("\n[4-5] Coverage CIs and conservative stratified conformal...")
    coverage = run_coverage_ci_and_conservative_stratified(
        train,
        calib,
        test,
        FULL_FEATURES,
        alpha=0.10,
    )
    results["diagnostics"]["coverage_ci_and_conservative_stratified"] = coverage
    print_coverage_table(coverage)

    out_path = RESULTS_DIR / "diagnostic_log.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Diagnostics saved to {out_path}")
    print("\n✓ Diagnostic suite complete.")

    return results


if __name__ == "__main__":
    run_all_diagnostics()