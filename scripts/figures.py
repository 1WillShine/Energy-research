"""
scripts/figures.py — Generate all paper figures from experiment results.

Run after experiments.py. All figures saved to figures/ directory.

Usage:
    python scripts/figures.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

FIGURES.mkdir(exist_ok=True)


# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


COLORS = {
    "global": "#2166AC",
    "stratified": "#D6604D",
    "nominal": "#1A9850",
    "heat": "#F46D43",
    "neutral": "#666666",
}

REGIME_COLORS = {
    "<75°F": "#4393C3",
    "75-90°F": "#92C5DE",
    "90-100°F": "#F4A582",
    ">100°F": "#D6604D",
}

TEMP_REGIMES = ["<75°F", "75-90°F", "90-100°F", ">100°F"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_results() -> dict:
    path = RESULTS / "experiment_log.json"

    if not path.exists():
        raise FileNotFoundError("Run scripts/experiments.py first.")

    with open(path) as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    """
    Robust project-loader helper.
    """
    try:
        from data.load_demand import load_processed
        return load_processed()
    except Exception:
        from data.load import load_processed
        return load_processed()


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        x = float(x)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def get_nested(dct, path, default=None):
    cur = dct

    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)

    return cur


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    print(f"  ✓ Figure saved: {path.name}")
    plt.close(fig)


def extract_coverage_cell_value(cell, prefer_gap=False):
    """
    Extract numeric value from any coverage-cell schema.

    Handles:
    - None
    - {"coverage": None}
    - {"coverage": ...}
    - {"stratified_coverage": ...}
    - {"coverage_gap_from_nominal": ...}
    """
    if not isinstance(cell, dict):
        return np.nan

    if prefer_gap:
        value = safe_float(cell.get("coverage_gap_from_nominal"))
        if np.isfinite(value):
            return value

    for key in [
        "coverage",
        "stratified_coverage",
        "observed_coverage",
        "global_coverage",
        "empirical_coverage",
    ]:
        value = safe_float(cell.get(key))
        if np.isfinite(value):
            return value

    return np.nan


def extract_n_spikes(cell):
    if not isinstance(cell, dict):
        return 0

    for key in ["n_spikes", "n", "count", "support"]:
        if key in cell:
            return safe_int(cell.get(key), 0)

    return 0


def is_fallback_cell(cell):
    return isinstance(cell, dict) and bool(cell.get("fallback", False))


def format_coverage_annotation(cell, value, prefer_gap=False):
    n = extract_n_spikes(cell)
    fallback = is_fallback_cell(cell)

    if not np.isfinite(value):
        return f"N/A\nn={n}"

    if prefer_gap:
        text = f"{value:+.2f}\nn={n}"
    else:
        text = f"{value:.2f}\nn={n}"

    if fallback:
        text += "*"

    return text


def maybe_float_format(x, digits=3, na="N/A"):
    value = safe_float(x)

    if not np.isfinite(value):
        return na

    return f"{value:.{digits}f}"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: COVERAGE DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_coverage_degradation(results: dict):
    """
    Main result: empirical coverage by temperature regime for standard vs.
    stratified conformal.
    """
    exp = results["experiments"]

    alpha = safe_float(exp["conformal_standard"].get("alpha"), default=0.10)
    nominal = 1.0 - alpha

    comparison = get_nested(
        exp,
        ["conformal_stratified", "coverage_comparison"],
        default={},
    )

    std_cov = []
    strat_cov = []
    n_spikes = []

    for r in TEMP_REGIMES:
        cell = comparison.get(r, {}) if isinstance(comparison, dict) else {}

        std_cov.append(safe_float(cell.get("global_coverage")))
        strat_cov.append(safe_float(cell.get("stratified_coverage")))
        n_spikes.append(safe_int(cell.get("n_spikes"), 0))

    x = np.arange(len(TEMP_REGIMES))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        std_cov,
        width,
        label="Standard conformal",
        color=COLORS["global"],
        alpha=0.85,
        edgecolor="white",
    )

    ax.bar(
        x + width / 2,
        strat_cov,
        width,
        label="Stratified conformal",
        color=COLORS["stratified"],
        alpha=0.85,
        edgecolor="white",
    )

    ax.axhline(
        nominal,
        color=COLORS["nominal"],
        linestyle="--",
        linewidth=1.8,
        label=f"Nominal coverage ({nominal:.0%})",
        zorder=5,
    )

    for i, n in enumerate(n_spikes):
        if n > 0:
            ax.text(
                x[i],
                0.02,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(TEMP_REGIMES)
    ax.set_xlabel("Temperature Regime", fontweight="bold")
    ax.set_ylabel("Empirical Coverage", fontweight="bold")
    ax.set_title(
        "Figure 1: Conformal Coverage Degrades Under High Temperature\n"
        "Standard conformal fails in high-heat regimes; stratified calibration restores coverage",
        fontsize=10,
        pad=12,
    )

    ax.set_ylim(0, 1.1)
    ax.axhspan(
        0,
        nominal - 0.05,
        alpha=0.04,
        color="red",
        label="Under-coverage zone",
    )
    ax.legend(loc="lower left", fontsize=9)

    save_figure(fig, FIGURES / "fig1_coverage_degradation.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CALIBRATION CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_calibration_curves(results: dict):
    """
    Reliability diagram.
    """
    clf = results["experiments"]["classifier"]

    train_cal = clf["train"].get("calibration_fraction_pos", [])
    train_pred = clf["train"].get("calibration_mean_pred", [])

    test_cal = clf["test"].get("calibration_fraction_pos", [])
    test_pred = clf["test"].get("calibration_mean_pred", [])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.2,
        label="Perfect calibration",
        zorder=1,
    )

    ax.plot(
        train_pred,
        train_cal,
        "o-",
        color=COLORS["global"],
        linewidth=2,
        markersize=6,
        label="Train 2021-22",
        alpha=0.9,
    )

    ax.plot(
        test_pred,
        test_cal,
        "s-",
        color=COLORS["stratified"],
        linewidth=2,
        markersize=6,
        label="Test 2024",
        alpha=0.9,
    )

    ax.set_xlabel("Mean Predicted Probability, spike", fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", fontweight="bold")
    ax.set_title(
        "Figure 2: Classifier Calibration Curve\n"
        "Deviation from diagonal = miscalibration",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    train_auroc = safe_float(clf["train"].get("auroc"))
    test_auroc = safe_float(clf["test"].get("auroc"))

    ax.text(
        0.98,
        0.05,
        f"Train AUROC = {maybe_float_format(train_auroc)}\n"
        f"Test AUROC  = {maybe_float_format(test_auroc)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
        ),
    )

    ax.legend(fontsize=9)

    save_figure(fig, FIGURES / "fig2_calibration_curves.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_feature_importance(results: dict):
    """
    Horizontal bar chart of GBM feature importances.
    """
    importances = get_nested(
        results,
        ["experiments", "classifier", "feature_importances"],
        default={},
    )

    if not importances:
        print("  ⚠ Figure 3: feature importances missing")
        return

    top_n = min(12, len(importances))

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_items)

    name_map = {
        "temp_f": "Temperature °F",
        "temp_f_sq": "Temperature²",
        "heat_stress": "Heat stress >95°F",
        "extreme_heat": "Extreme heat",
        "temp_x_peak_hour": "Temp × peak hour",
        "hour": "Hour of day",
        "solar_rad": "Solar radiation",
        "wind_mph": "Wind speed",
        "month": "Month",
        "dow": "Day of week",
        "is_summer": "Summer indicator",
        "is_weekend": "Weekend indicator",
    }

    clean_names = [name_map.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [
        COLORS["heat"] if ("temp" in f.lower() or "heat" in f.lower())
        else COLORS["global"]
        for f in features
    ]

    y_pos = np.arange(top_n)

    ax.barh(
        y_pos,
        values,
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=9.5)
    ax.invert_yaxis()

    ax.set_xlabel("Feature Importance, GBM", fontweight="bold")
    ax.set_title(
        "Figure 3: Feature Importances for Weather-Only Spike Classifier\n"
        "Orange = weather/temperature features",
        fontsize=10,
    )

    for i, v in enumerate(values):
        ax.text(
            float(v) + 0.001,
            i,
            f"{float(v):.3f}",
            va="center",
            fontsize=8,
        )

    save_figure(fig, FIGURES / "fig3_feature_importance.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: HEAT WAVE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_heat_wave_timeline(df: pd.DataFrame, results: dict):
    """
    Timeline figure for September 2022 heat wave.
    """
    if "datetime" not in df.columns:
        print("  ⚠ Figure 4: datetime column missing")
        return

    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

    required = ["datetime", "temp_f", "spike"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"  ⚠ Figure 4: missing columns {missing}")
        return

    price_col = "lmp" if "lmp" in df.columns else "demand_mw" if "demand_mw" in df.columns else None

    if price_col is None:
        print("  ⚠ Figure 4: neither lmp nor demand_mw available")
        return

    heat_period = df[
        (df["datetime"].dt.year == 2022)
        & (df["datetime"].dt.month == 9)
        & (df["datetime"].dt.day <= 15)
    ].copy().sort_values("datetime")

    if len(heat_period) == 0:
        print("  ⚠ Figure 4: Sept 2022 data not in dataset")
        return

    ylabel = "LMP ($/MWh)" if price_col == "lmp" else "Demand MW"
    label = "LMP ($/MWh)" if price_col == "lmp" else "Demand MW"

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(
        heat_period["datetime"],
        heat_period[price_col],
        color=COLORS["global"],
        linewidth=1.2,
        label=label,
        zorder=3,
    )

    spikes = heat_period[heat_period["spike"] == 1]

    ax1.scatter(
        spikes["datetime"],
        spikes[price_col],
        color=COLORS["heat"],
        s=20,
        zorder=4,
        label="Spike top 5%",
        alpha=0.8,
    )

    threshold_price = heat_period[price_col].quantile(0.95)

    ax1.axhline(
        threshold_price,
        color=COLORS["heat"],
        linestyle=":",
        alpha=0.6,
        label=f"95th percentile ({threshold_price:.0f})",
    )

    ax1.set_ylabel(ylabel, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.set_title(
        "Figure 4: California Grid Stress and Temperature — September 2022 Heat Wave\n"
        "Extreme temperatures coincide with spike events outside normal conditions",
        fontsize=10,
    )

    ax2.fill_between(
        heat_period["datetime"],
        heat_period["temp_f"],
        alpha=0.4,
        color=COLORS["heat"],
    )

    ax2.plot(
        heat_period["datetime"],
        heat_period["temp_f"],
        color=COLORS["heat"],
        linewidth=1.5,
        label="Temperature °F",
    )

    ax2.axhline(
        95,
        color="darkred",
        linestyle="--",
        linewidth=1,
        label="95°F heat stress",
    )

    ax2.axhline(
        105,
        color="darkred",
        linestyle=":",
        linewidth=1,
        label="105°F extreme heat",
    )

    ax2.set_xlabel("Date", fontweight="bold")
    ax2.set_ylabel("Temperature °F", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left")

    # Handle timezone-aware or naive datetimes.
    tz = getattr(heat_period["datetime"].dt, "tz", None)

    if tz is not None:
        start = pd.Timestamp("2022-09-05", tz=tz)
        end = pd.Timestamp("2022-09-09", tz=tz)
    else:
        start = pd.Timestamp("2022-09-05")
        end = pd.Timestamp("2022-09-09")

    for ax in [ax1, ax2]:
        ax.axvspan(
            start,
            end,
            alpha=0.12,
            color="red",
            label="_nolegend_",
        )

    save_figure(fig, FIGURES / "fig4_heatwave_timeline.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: COVARIATE SHIFT
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_covariate_shift(df: pd.DataFrame, results: dict):
    """
    Temperature distribution shift between train and test.
    """
    required = ["year", "temp_f"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠ Figure 5: missing columns {missing}")
        return

    train_temp = df[df["year"].isin([2021, 2022])]["temp_f"].dropna()
    test_temp = df[df["year"] == 2024]["temp_f"].dropna()

    if len(train_temp) == 0 or len(test_temp) == 0:
        print("  ⚠ Figure 5: insufficient train/test temperature data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bins = np.linspace(
        min(train_temp.min(), test_temp.min()),
        max(train_temp.max(), test_temp.max()),
        40,
    )

    ax1.hist(
        train_temp,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["global"],
        label="Train 2021-22",
    )

    ax1.hist(
        test_temp,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["stratified"],
        label="Test 2024",
    )

    ax1.axvline(
        95,
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label="95°F threshold",
    )

    ax1.set_xlabel("Temperature °F", fontweight="bold")
    ax1.set_ylabel("Density", fontweight="bold")
    ax1.set_title("Temperature Distribution Shift\nTrain vs. Test", fontsize=10)
    ax1.legend(fontsize=9)

    shift = get_nested(
        results,
        ["experiments", "covariate_shift"],
        default={},
    )

    auroc = safe_float(shift.get("auroc_mean"), default=np.nan)
    std = safe_float(shift.get("auroc_std"), default=0.0)
    interpretation = shift.get("interpretation", "Unavailable")

    if np.isfinite(auroc):
        ax2.barh(
            ["Shift detector AUROC"],
            [auroc],
            xerr=[std],
            color=COLORS["global"] if auroc < 0.7 else COLORS["stratified"],
            alpha=0.85,
            capsize=6,
            height=0.4,
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "Shift detector unavailable",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.axvline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Random 0.5",
    )

    ax2.axvline(
        0.7,
        color="darkred",
        linestyle=":",
        linewidth=1.2,
        label="Strong shift 0.7",
    )

    ax2.set_xlim(0.4, 1.0)
    ax2.set_xlabel("AUROC, weather features only", fontweight="bold")
    ax2.set_title(
        f"Classifier-Based Shift Detection\n"
        f"AUROC = {maybe_float_format(auroc)} ± {maybe_float_format(std)} → {interpretation}",
        fontsize=10,
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Figure 5: Covariate Shift in Weather Features, 2021-22 → 2024",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    save_figure(fig, FIGURES / "fig5_covariate_shift.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: VULNERABILITY COVERAGE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_vulnerability_coverage_matrix(results: dict):
    """
    Equity figure: vulnerability level × temperature regime → coverage.

    Robust to:
      - None cells
      - fallback cells
      - sparse cells
      - missing groups/regimes
      - mixed schemas
    """
    vuln_exp = get_nested(
        results,
        ["experiments", "vulnerability"],
        default={},
    )

    if not vuln_exp:
        print("  ⚠ Figure 6: vulnerability experiment not found")
        return

    if "error" in vuln_exp and not vuln_exp.get("coverage_matrix"):
        print(f"  ⚠ Figure 6: {vuln_exp['error']}")
        return

    matrix = vuln_exp.get("coverage_matrix", {})

    if not isinstance(matrix, dict) or not matrix:
        print("  ⚠ Figure 6: coverage_matrix missing or empty")
        return

    alpha = safe_float(
        get_nested(results, ["experiments", "conformal_standard", "alpha"], default=0.10),
        default=0.10,
    )
    nominal = 1.0 - alpha

    # Keep only rows that look like vulnerability groups.
    vuln_labels = [
        k for k, v in matrix.items()
        if isinstance(v, dict) and any(regime in v for regime in TEMP_REGIMES)
    ]

    preferred_order = [
        "Low vulnerability",
        "Medium vulnerability",
        "High vulnerability",
        "Low",
        "Medium",
        "High",
    ]

    ordered = [x for x in preferred_order if x in vuln_labels]
    ordered += [x for x in vuln_labels if x not in ordered]
    vuln_labels = ordered

    if not vuln_labels:
        print("  ⚠ Figure 6: no vulnerability groups found")
        return

    n_rows = len(vuln_labels)

    coverage_grid = np.full((n_rows, len(TEMP_REGIMES)), np.nan)
    gap_grid = np.full((n_rows, len(TEMP_REGIMES)), np.nan)
    n_grid = np.zeros((n_rows, len(TEMP_REGIMES)), dtype=int)
    fallback_grid = np.zeros((n_rows, len(TEMP_REGIMES)), dtype=bool)

    for vi, vl in enumerate(vuln_labels):
        row = matrix.get(vl, {})

        for ri, regime in enumerate(TEMP_REGIMES):
            cell = row.get(regime)

            cov = extract_coverage_cell_value(cell, prefer_gap=False)

            if np.isfinite(cov):
                coverage_grid[vi, ri] = cov
                gap_grid[vi, ri] = cov - nominal

            gap = extract_coverage_cell_value(cell, prefer_gap=True)
            if np.isfinite(gap):
                gap_grid[vi, ri] = gap

            n_grid[vi, ri] = extract_n_spikes(cell)
            fallback_grid[vi, ri] = is_fallback_cell(cell)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    masked_gap = np.ma.masked_invalid(gap_grid)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(
        masked_gap,
        cmap=cmap,
        vmin=-0.5,
        vmax=0.2,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(TEMP_REGIMES)))
    ax.set_xticklabels(TEMP_REGIMES, fontsize=9.5)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(vuln_labels, fontsize=10)

    ax.set_xlabel("Temperature Regime", fontweight="bold")
    ax.set_ylabel("Community Vulnerability Group", fontweight="bold")

    ax.set_title(
        "Figure 6: Coverage Degradation by Vulnerability × Temperature Regime\n"
        f"Nominal coverage = {nominal:.0%}. Red = under-coverage. "
        "Asterisk = fallback for sparse subgroup cell.",
        fontsize=10,
        fontweight="bold",
        pad=12,
    )

    for vi in range(n_rows):
        for ri in range(len(TEMP_REGIMES)):
            gap = gap_grid[vi, ri]
            cov = coverage_grid[vi, ri]
            n = n_grid[vi, ri]
            fb = fallback_grid[vi, ri]

            if np.isfinite(cov):
                label = f"{cov:.2f}\nΔ={gap:+.2f}\nn={n}"
                if fb:
                    label += "*"

                text_color = "white" if gap < -0.25 else "black"
                fontweight = "normal" if fb else "bold"
                fontstyle = "italic" if fb else "normal"
            else:
                label = f"N/A\nn={n}"
                text_color = "gray"
                fontweight = "normal"
                fontstyle = "italic"

            ax.text(
                ri,
                vi,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight=fontweight,
                fontstyle=fontstyle,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Coverage − Nominal")

    fig.text(
        0.01,
        0.01,
        "* fallback cell: subgroup-temperature spike count below threshold; "
        "displayed value uses broader fallback coverage.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = FIGURES / "fig6_vulnerability_coverage_matrix.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))

    print(f"  ✓ Figure saved: {path.name}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: VULNERABILITY BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_vulnerability_map(results: dict):
    """
    Portable substitute for map: county-level energy burden bar chart.
    """
    vuln_exp = get_nested(
        results,
        ["experiments", "vulnerability"],
        default={},
    )

    if not vuln_exp or "error" in vuln_exp:
        print("  ⚠ Figure 7: vulnerability data unavailable")
        return

    try:
        from data.vulnerability import load_vulnerability
        vuln_df = load_vulnerability()
    except Exception as e:
        print(f"  ⚠ Figure 7: could not load vulnerability data: {e}")
        return

    required = ["county_name", "energy_burden_pct", "energy_burden_quintile"]
    missing = [c for c in required if c not in vuln_df.columns]

    if missing:
        print(f"  ⚠ Figure 7: missing columns {missing}")
        return

    vuln_df = vuln_df.sort_values("energy_burden_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [
        REGIME_COLORS[">100°F"] if q == 5
        else REGIME_COLORS["90-100°F"] if q in [3, 4]
        else REGIME_COLORS["75-90°F"] if q == 2
        else REGIME_COLORS["<75°F"]
        for q in vuln_df["energy_burden_quintile"]
    ]

    ax.barh(
        range(len(vuln_df)),
        vuln_df["energy_burden_pct"],
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )

    ax.set_yticks(range(len(vuln_df)))
    ax.set_yticklabels(vuln_df["county_name"], fontsize=9)

    mean_burden = vuln_df["energy_burden_pct"].mean()

    ax.axvline(
        mean_burden,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"CA mean ({mean_burden:.1f}%)",
    )

    ax.set_xlabel("Energy Burden, % of household income", fontweight="bold")
    ax.set_title(
        "Figure 7: Energy Burden by California County\n"
        "Darker = higher vulnerability quintile",
        fontsize=10,
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=REGIME_COLORS["<75°F"], label="Q1 lowest burden"),
        Patch(facecolor=REGIME_COLORS["75-90°F"], label="Q2"),
        Patch(facecolor=REGIME_COLORS["90-100°F"], label="Q3-Q4"),
        Patch(facecolor=REGIME_COLORS[">100°F"], label="Q5 highest burden"),
    ]

    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    save_figure(fig, FIGURES / "fig7_vulnerability_map.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: dict):
    """
    Print text/LaTeX-friendly summary.
    """
    exp = results["experiments"]

    alpha = safe_float(
        get_nested(exp, ["conformal_standard", "alpha"], default=0.10),
        default=0.10,
    )
    nominal = 1.0 - alpha

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    clf = exp.get("classifier", {})

    train_auroc = get_nested(clf, ["train", "auroc"], default=None)
    train_brier = get_nested(clf, ["train", "brier"], default=None)
    test_auroc = get_nested(clf, ["test", "auroc"], default=None)
    test_brier = get_nested(clf, ["test", "brier"], default=None)

    print("\nClassifier Performance:")
    print(
        f"  Train AUROC: {maybe_float_format(train_auroc)} "
        f"| Brier: {maybe_float_format(train_brier, 4)}"
    )
    print(
        f"  Test  AUROC: {maybe_float_format(test_auroc)} "
        f"| Brier: {maybe_float_format(test_brier, 4)}"
    )

    comparison = get_nested(
        exp,
        ["conformal_stratified", "coverage_comparison"],
        default={},
    )

    print(f"\nConformal Coverage nominal={nominal:.0%}, alpha={alpha}:")
    print(f"{'Regime':<14} {'Global':>10} {'Stratified':>12} {'Delta':>8} {'n':>6}")
    print("-" * 58)

    if isinstance(comparison, dict):
        for regime in TEMP_REGIMES:
            data = comparison.get(regime, {})

            if not isinstance(data, dict):
                continue

            global_cov = safe_float(data.get("global_coverage"))
            strat_cov = safe_float(data.get("stratified_coverage"))
            delta = safe_float(data.get("delta"))
            n = safe_int(data.get("n_spikes"), 0)

            print(
                f"{regime:<14} "
                f"{maybe_float_format(global_cov):>10} "
                f"{maybe_float_format(strat_cov):>12} "
                f"{maybe_float_format(delta):>8} "
                f"{n:>6}"
            )

    overall = get_nested(
        exp,
        ["conformal_standard", "coverage", "empirical_coverage"],
        default=None,
    )

    print(
        f"\nOverall empirical coverage: "
        f"{maybe_float_format(overall)} nominal={nominal:.2f}"
    )

    shift = exp.get("covariate_shift", {})
    auroc = shift.get("auroc_mean")
    std = shift.get("auroc_std")

    print("\nCovariate Shift weather features, 2021-22 → 2024:")
    print(f"  AUROC = {maybe_float_format(auroc)} ± {maybe_float_format(std)}")
    print(f"  Interpretation: {shift.get('interpretation', 'N/A')}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_figures():
    print("Loading results and data...")

    results = load_results()

    try:
        df = load_data()
        has_data = True
    except Exception as e:
        print(f"  ⚠ Processed data not available — skipping data-dependent figures: {e}")
        df = None
        has_data = False

    print("\nGenerating figures...")

    fig1_coverage_degradation(results)
    fig2_calibration_curves(results)
    fig3_feature_importance(results)

    if has_data:
        fig4_heat_wave_timeline(df, results)
        fig5_covariate_shift(df, results)

    fig6_vulnerability_coverage_matrix(results)
    fig7_vulnerability_map(results)

    print_results_table(results)

    print(f"\n✓ All figures saved to {FIGURES}/")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    generate_all_figures()