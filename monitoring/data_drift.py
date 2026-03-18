"""
monitoring/data_drift.py — Feature distribution drift detection

Compares reference (training) data against current (recent) data using:
  - Kolmogorov-Smirnov (KS) test for continuous features
  - Population Stability Index (PSI) for critical features
  - Chi-squared test for categorical features

Outputs a drift report and exits with code 1 if drift is detected.

Usage:
  python monitoring/data_drift.py \\
    --reference data/port_calls.parquet \\
    --current   data/recent_30days.parquet \\
    --report    monitoring/drift_report.json

  # Or using date range from the main parquet
  python monitoring/data_drift.py \\
    --reference data/port_calls.parquet \\
    --current-days 30
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature groups ────────────────────────────────────────────────────────────

CONTINUOUS_FEATURES = [
    "waiting_anchor_hours",
    "berth_competition",
    "weather_wind_knots",
    "teu_capacity",
    "dwt",
    "loa",
    "cargo_tons",
]

CATEGORICAL_FEATURES = [
    "vessel_type",
    "port_name",
    "company_name",
]

# Features that trigger alerts if KS p < threshold
CRITICAL_FEATURES = [
    "waiting_anchor_hours",
    "berth_competition",
    "weather_wind_knots",
]

KS_ALERT_THRESHOLD  = 0.05   # p-value
PSI_ALERT_THRESHOLD = 0.25   # PSI score (0.1=minor, 0.2=moderate, 0.25=major)
N_BINS              = 10     # for PSI calculation


# ─────────────────────────────────────────────────────────────────────────────
# PSI
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.1: No significant change.
    PSI 0.1-0.2: Moderate change — investigate.
    PSI > 0.2: Significant change — action required.
    """
    # Define bins on reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  = -np.inf
    bins[-1] =  np.inf

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current,   bins=bins)

    ref_pct = (ref_counts + 0.0001) / len(reference)
    cur_pct = (cur_counts + 0.0001) / len(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────────
# KS TEST
# ─────────────────────────────────────────────────────────────────────────────

def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Returns (ks_statistic, p_value). Low p = distributions differ."""
    stat, pval = stats.ks_2samp(reference, current)
    return float(stat), float(pval)


# ─────────────────────────────────────────────────────────────────────────────
# CHI-SQUARED TEST (categoricals)
# ─────────────────────────────────────────────────────────────────────────────

def chi2_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Returns (chi2_statistic, p_value)."""
    all_cats = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(all_cats, fill_value=0).values + 1
    cur_counts = current.value_counts().reindex(all_cats, fill_value=0).values + 1
    # Scale to same total
    cur_expected = cur_counts / cur_counts.sum() * ref_counts.sum()
    stat, pval = stats.chisquare(f_obs=ref_counts, f_exp=cur_expected)
    return float(stat), float(pval)


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATS
# ─────────────────────────────────────────────────────────────────────────────

def describe(arr: np.ndarray) -> dict:
    return {
        "mean":   round(float(np.mean(arr)), 3),
        "std":    round(float(np.std(arr)),  3),
        "median": round(float(np.median(arr)), 3),
        "p80":    round(float(np.percentile(arr, 80)), 3),
        "p95":    round(float(np.percentile(arr, 95)), 3),
        "min":    round(float(np.min(arr)),  3),
        "max":    round(float(np.max(arr)),  3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DRIFT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_drift_check(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    ks_threshold: float = KS_ALERT_THRESHOLD,
    psi_threshold: float = PSI_ALERT_THRESHOLD,
) -> dict:
    """
    Run full drift analysis. Returns report dict.
    """
    report = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "reference_rows": len(ref_df),
        "current_rows":   len(cur_df),
        "reference_period": {
            "start": str(ref_df["ata_actual"].min().date()) if "ata_actual" in ref_df else "N/A",
            "end":   str(ref_df["ata_actual"].max().date()) if "ata_actual" in ref_df else "N/A",
        },
        "current_period": {
            "start": str(cur_df["ata_actual"].min().date()) if "ata_actual" in cur_df else "N/A",
            "end":   str(cur_df["ata_actual"].max().date()) if "ata_actual" in cur_df else "N/A",
        },
        "continuous_features": {},
        "categorical_features": {},
        "alerts": [],
        "drift_detected": False,
        "critical_drift": False,
    }

    log.info("Checking %d continuous features...", len(CONTINUOUS_FEATURES))

    # ── Continuous features ───────────────────────────────────────────────────
    for feat in CONTINUOUS_FEATURES:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            log.warning("  Feature not found: %s", feat)
            continue

        ref_arr = ref_df[feat].dropna().values
        cur_arr = cur_df[feat].dropna().values

        if len(ref_arr) < 30 or len(cur_arr) < 30:
            log.warning("  Insufficient data for %s", feat)
            continue

        ks_stat, ks_pval = ks_test(ref_arr, cur_arr)
        psi_val          = compute_psi(ref_arr, cur_arr)

        result = {
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value":   round(ks_pval, 6),
            "psi":          round(psi_val, 4),
            "drift_flag":   ks_pval < ks_threshold or psi_val > psi_threshold,
            "reference_stats": describe(ref_arr),
            "current_stats":   describe(cur_arr),
        }
        report["continuous_features"][feat] = result

        drift_label = "DRIFT" if result["drift_flag"] else "OK"
        log.info("  %-35s KS_p=%.4f  PSI=%.3f  [%s]",
                 feat, ks_pval, psi_val, drift_label)

        if result["drift_flag"]:
            severity = "CRITICAL" if feat in CRITICAL_FEATURES else "WARNING"
            report["alerts"].append({
                "feature":  feat,
                "severity": severity,
                "ks_p":     round(ks_pval, 6),
                "psi":      round(psi_val, 4),
                "message":  (
                    f"{feat}: {'KS p=' + str(round(ks_pval,4)) if ks_pval < ks_threshold else ''}"
                    f"{'  PSI=' + str(round(psi_val,3)) if psi_val > psi_threshold else ''}"
                ),
            })

    # ── Categorical features ──────────────────────────────────────────────────
    log.info("Checking %d categorical features...", len(CATEGORICAL_FEATURES))
    for feat in CATEGORICAL_FEATURES:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue

        chi2_stat, chi2_pval = chi2_test(ref_df[feat].dropna(), cur_df[feat].dropna())

        ref_dist = ref_df[feat].value_counts(normalize=True).to_dict()
        cur_dist = cur_df[feat].value_counts(normalize=True).to_dict()

        result = {
            "chi2_statistic": round(chi2_stat, 3),
            "chi2_p_value":   round(chi2_pval, 6),
            "drift_flag":     chi2_pval < ks_threshold,
            "reference_distribution": {k: round(v, 3) for k, v in ref_dist.items()},
            "current_distribution":   {k: round(v, 3) for k, v in cur_dist.items()},
        }
        report["categorical_features"][feat] = result

        drift_label = "DRIFT" if result["drift_flag"] else "OK"
        log.info("  %-35s chi2_p=%.4f  [%s]", feat, chi2_pval, drift_label)

        if result["drift_flag"]:
            report["alerts"].append({
                "feature":  feat,
                "severity": "WARNING",
                "chi2_p":   round(chi2_pval, 6),
                "message":  f"{feat}: distribution shift detected (chi2 p={chi2_pval:.4f})",
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_alerts  = len(report["alerts"])
    n_critical = sum(1 for a in report["alerts"] if a["severity"] == "CRITICAL")
    report["drift_detected"] = n_alerts > 0
    report["critical_drift"] = n_critical >= 2   # 2+ critical features drifted

    log.info("\n=== DRIFT SUMMARY ===")
    log.info("  Total alerts:    %d", n_alerts)
    log.info("  Critical alerts: %d", n_critical)
    log.info("  Drift detected:  %s", report["drift_detected"])
    log.info("  Critical drift:  %s", report["critical_drift"])

    if report["critical_drift"]:
        log.warning("  ACTION REQUIRED: Trigger retraining pipeline.")
        log.warning("  Run: python retrain.py --input data/port_calls.parquet --validate")
    elif report["drift_detected"]:
        log.info("  MONITOR: Some features drifting. Review before next scheduled retrain.")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature drift detection")
    parser.add_argument("--reference",    required=True, help="Reference parquet/CSV (training data)")
    parser.add_argument("--current",      help="Current window parquet/CSV")
    parser.add_argument("--current-days", type=int, default=30,
                        help="If --current not provided, use last N days of reference")
    parser.add_argument("--report",       default="monitoring/drift_report.json",
                        help="Output JSON report path")
    parser.add_argument("--ks-threshold",  type=float, default=KS_ALERT_THRESHOLD)
    parser.add_argument("--psi-threshold", type=float, default=PSI_ALERT_THRESHOLD)
    parser.add_argument("--exit-on-drift", action="store_true",
                        help="Exit with code 1 if critical drift detected")
    args = parser.parse_args()

    # Load reference
    ref_df = pd.read_parquet(args.reference) if args.reference.endswith(".parquet") \
             else pd.read_csv(args.reference)
    if "ata_actual" in ref_df.columns:
        ref_df["ata_actual"] = pd.to_datetime(ref_df["ata_actual"])

    # Load or slice current
    if args.current:
        cur_df = pd.read_parquet(args.current) if args.current.endswith(".parquet") \
                 else pd.read_csv(args.current)
        if "ata_actual" in cur_df.columns:
            cur_df["ata_actual"] = pd.to_datetime(cur_df["ata_actual"])
    else:
        cutoff = ref_df["ata_actual"].max() - timedelta(days=args.current_days)
        ref_cutoff = ref_df["ata_actual"].max() - timedelta(days=args.current_days * 4)
        cur_df  = ref_df[ref_df["ata_actual"] >= cutoff].copy()
        ref_df  = ref_df[ref_df["ata_actual"] <  cutoff].copy()
        log.info("Current window: last %d days  |  Reference: before that", args.current_days)

    log.info("Reference: %d rows  |  Current: %d rows", len(ref_df), len(cur_df))

    if len(cur_df) < 100:
        log.error("Current window too small (%d rows). Need >= 100.", len(cur_df))
        sys.exit(1)

    # Run drift check
    report = run_drift_check(
        ref_df, cur_df,
        ks_threshold=args.ks_threshold,
        psi_threshold=args.psi_threshold,
    )

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report saved: %s", report_path)

    if args.exit_on_drift and report["critical_drift"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
