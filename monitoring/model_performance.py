"""
monitoring/model_performance.py — Model accuracy degradation detection

Compares model predictions against actual waiting times (backfilled from TOS)
and tracks rolling performance metrics over time.

Metrics tracked:
  - Model 1 (waiting time): MAE, MAPE, R²  — rolling 7-day and overall
  - Model 3 (congestion): Precision, Recall, F1, AUC  — rolling 7-day
  - Model 2 (occupancy): Accuracy  — rolling 7-day

Alerts if:
  - Rolling 7-day MAE > 4h (2x the 1.22h training MAE; alert at 2x for practical operations)
  - Rolling 7-day congestion precision < 0.70 (below operational threshold)
  - Rolling 7-day congestion recall < 0.60 (missing too many real events)

Usage:
  python monitoring/model_performance.py \\
    --predictions  monitoring/predictions_log.csv \\
    --report       monitoring/performance_report.json

  # Or against PostgreSQL
  python monitoring/model_performance.py \\
    --db-url  postgresql://portuser:portpass@localhost:5432/port_intelligence \\
    --report  monitoring/performance_report.json

  # Minimum example (CSV with required columns):
  # ata_actual, waiting_anchor_hours_actual, waiting_anchor_forecast,
  # congestion_flag_actual, congestion_flag_predicted, congestion_score,
  # occupancy_class_actual, occupancy_class_predicted
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

# Optional: sklearn for AUC
try:
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score,
        mean_absolute_error, r2_score, accuracy_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────

TRAIN_MAE      = 1.22   # from model card
TRAIN_MAPE     = 17.7   # %
TRAIN_PRECISION = 0.969
TRAIN_RECALL   = 0.802
TRAIN_AUC      = 0.991
TRAIN_OCC_ACC  = 0.85   # expected real-data lower bound from model card

MAE_ALERT_THRESHOLD       = 4.0    # h   — CRITICAL if exceeded
MAE_WARN_THRESHOLD        = 2.5    # h   — WARNING if exceeded
PRECISION_ALERT_THRESHOLD = 0.70   # below this → CRITICAL
RECALL_ALERT_THRESHOLD    = 0.60   # below this → CRITICAL
AUC_WARN_THRESHOLD        = 0.85   # below this → WARNING
OCC_ALERT_THRESHOLD       = 0.65   # below this → WARNING

ROLLING_WINDOW_DAYS = 7
MAPE_MIN_HOURS      = 3.0   # same as train_models.py — exclude near-zero actuals


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mape(actual: np.ndarray, predicted: np.ndarray, min_val: float = MAPE_MIN_HOURS) -> float:
    """MAPE on rows where actual >= min_val (avoids near-zero instability)."""
    mask = actual >= min_val
    if mask.sum() < 5:
        return float("nan")
    a = actual[mask]
    p = predicted[mask]
    return float(np.mean(np.abs((a - p) / a)) * 100)


def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(actual)
    if n < 10:
        return {"n": n, "mae": None, "mape": None, "r2": None, "rmse": None}
    mae   = float(mean_absolute_error(actual, predicted)) if _SKLEARN else float(np.mean(np.abs(actual - predicted)))
    mape  = compute_mape(actual, predicted)
    r2    = float(r2_score(actual, predicted)) if _SKLEARN else None
    rmse  = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"n": n, "mae": round(mae, 3), "mape": round(mape, 2) if not np.isnan(mape) else None,
            "r2": round(r2, 4) if r2 is not None else None, "rmse": round(rmse, 3)}


def compute_classification_metrics(actual: np.ndarray, predicted: np.ndarray,
                                   scores=None) -> dict:
    n = len(actual)
    if n < 10:
        return {"n": n, "precision": None, "recall": None, "f1": None, "auc": None}

    if _SKLEARN:
        prec = float(precision_score(actual, predicted, zero_division=0))
        rec  = float(recall_score(actual, predicted, zero_division=0))
        f1   = float(f1_score(actual, predicted, zero_division=0))
        auc  = float(roc_auc_score(actual, scores)) if scores is not None and len(np.unique(actual)) > 1 else None
    else:
        tp = int(np.sum((actual == 1) & (predicted == 1)))
        fp = int(np.sum((actual == 0) & (predicted == 1)))
        fn = int(np.sum((actual == 1) & (predicted == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        auc  = None

    return {
        "n":         n,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "auc":       round(auc, 4) if auc is not None else None,
    }


def compute_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(actual)
    if n < 10:
        return {"n": n, "accuracy": None}
    acc = float(accuracy_score(actual, predicted)) if _SKLEARN else float(np.mean(actual == predicted))
    return {"n": n, "accuracy": round(acc, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING WINDOW ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mae_series(df: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """
    Returns daily rolling MAE for Model 1.
    df must have: ata_actual (datetime), waiting_anchor_hours_actual, waiting_anchor_forecast
    """
    df = df.dropna(subset=["waiting_anchor_hours_actual", "waiting_anchor_forecast"]).copy()
    df = df.sort_values("ata_actual")
    df["date"] = df["ata_actual"].dt.date

    records = []
    unique_dates = sorted(df["date"].unique())
    for i, d in enumerate(unique_dates):
        start = d - timedelta(days=window_days - 1)
        mask = (df["date"] >= start) & (df["date"] <= d)
        window = df.loc[mask]
        if len(window) < 10:
            continue
        mae  = float(np.mean(np.abs(window["waiting_anchor_hours_actual"] - window["waiting_anchor_forecast"])))
        mape = compute_mape(
            window["waiting_anchor_hours_actual"].values,
            window["waiting_anchor_forecast"].values,
        )
        records.append({
            "date":  str(d),
            "n":     len(window),
            "mae":   round(mae, 3),
            "mape":  round(mape, 2) if not np.isnan(mape) else None,
        })
    return pd.DataFrame(records)


def rolling_congestion_series(df: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """
    Returns daily rolling congestion precision/recall for Model 3.
    df must have: ata_actual, congestion_flag_actual, congestion_flag_predicted
    """
    required = ["congestion_flag_actual", "congestion_flag_predicted"]
    df = df.dropna(subset=required).copy()
    df = df.sort_values("ata_actual")
    df["date"] = df["ata_actual"].dt.date

    records = []
    for d in sorted(df["date"].unique()):
        start = d - timedelta(days=window_days - 1)
        mask = (df["date"] >= start) & (df["date"] <= d)
        window = df.loc[mask]
        if len(window) < 10 or window["congestion_flag_actual"].sum() < 3:
            continue
        actual    = window["congestion_flag_actual"].values.astype(int)
        predicted = window["congestion_flag_predicted"].values.astype(int)
        scores    = window["congestion_score"].values if "congestion_score" in window.columns else None
        m = compute_classification_metrics(actual, predicted, scores)
        m["date"] = str(d)
        records.append(m)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ata_actual" in df.columns:
        df["ata_actual"] = pd.to_datetime(df["ata_actual"])
    return df


def load_from_db(db_url: str, days: int = 90) -> pd.DataFrame:
    """
    Load from vessel_predictions table joined with actuals.
    Requires SQLAlchemy.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        log.error("sqlalchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")
        sys.exit(1)

    engine = create_engine(db_url)
    cutoff = datetime.utcnow() - timedelta(days=days)
    query = f"""
        SELECT
            prediction_time           AS ata_actual,
            waiting_time_forecast     AS waiting_anchor_forecast,
            actual_waiting_time       AS waiting_anchor_hours_actual,
            congestion_predicted      AS congestion_flag_predicted,
            congestion_actual         AS congestion_flag_actual,
            congestion_score,
            occupancy_predicted       AS occupancy_class_predicted,
            occupancy_actual          AS occupancy_class_actual
        FROM vessel_predictions
        WHERE prediction_time >= '{cutoff.isoformat()}'
          AND actual_waiting_time IS NOT NULL
        ORDER BY prediction_time
    """
    df = pd.read_sql(query, engine)
    df["ata_actual"] = pd.to_datetime(df["ata_actual"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_performance_check(df: pd.DataFrame) -> dict:
    """
    Full performance analysis. Returns report dict with alerts.
    """
    n_total = len(df)
    has_actuals = df["waiting_anchor_hours_actual"].notna().sum() if "waiting_anchor_hours_actual" in df else 0

    log.info("Total prediction records: %d  |  With actuals: %d", n_total, has_actuals)

    report = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "total_records":  n_total,
        "records_with_actuals": int(has_actuals),
        "evaluation_window": {
            "start": str(df["ata_actual"].min().date()) if "ata_actual" in df else "N/A",
            "end":   str(df["ata_actual"].max().date()) if "ata_actual" in df else "N/A",
        },
        "model1_waiting_time": {},
        "model2_occupancy":    {},
        "model3_congestion":   {},
        "rolling_mae_by_day":        [],
        "rolling_congestion_by_day": [],
        "alerts":    [],
        "degradation_detected": False,
        "critical_degradation": False,
        "training_baselines": {
            "model1_mae":       TRAIN_MAE,
            "model1_mape_pct":  TRAIN_MAPE,
            "model3_precision": TRAIN_PRECISION,
            "model3_recall":    TRAIN_RECALL,
            "model3_auc":       TRAIN_AUC,
            "model2_accuracy":  TRAIN_OCC_ACC,
        },
    }

    # ── Model 1: Waiting time ─────────────────────────────────────────────────
    m1_cols = ["waiting_anchor_hours_actual", "waiting_anchor_forecast"]
    m1_df   = df.dropna(subset=m1_cols) if all(c in df.columns for c in m1_cols) else pd.DataFrame()

    if len(m1_df) >= 10:
        log.info("Model 1 (waiting time): %d records with actuals", len(m1_df))
        m1_metrics = compute_regression_metrics(
            m1_df["waiting_anchor_hours_actual"].values,
            m1_df["waiting_anchor_forecast"].values,
        )
        m1_metrics["vs_training"] = {
            "mae_delta":  round(m1_metrics["mae"] - TRAIN_MAE, 3) if m1_metrics["mae"] else None,
            "mape_delta": round(m1_metrics["mape"] - TRAIN_MAPE, 2) if m1_metrics["mape"] else None,
        }
        report["model1_waiting_time"] = m1_metrics

        log.info("  MAE=%.2fh  MAPE=%.1f%%  R2=%.3f  (training: MAE=%.2fh)",
                 m1_metrics["mae"], m1_metrics["mape"] or 0, m1_metrics["r2"] or 0, TRAIN_MAE)

        # Rolling series
        if "ata_actual" in m1_df.columns:
            rolling = rolling_mae_series(m1_df)
            report["rolling_mae_by_day"] = rolling.to_dict(orient="records")

            # Alert on most recent window
            if len(rolling) > 0:
                latest_mae = rolling.iloc[-1]["mae"]
                latest_date = rolling.iloc[-1]["date"]
                if latest_mae > MAE_ALERT_THRESHOLD:
                    report["alerts"].append({
                        "model":    "model1_waiting_time",
                        "severity": "CRITICAL",
                        "metric":   "rolling_7d_mae",
                        "value":    latest_mae,
                        "threshold": MAE_ALERT_THRESHOLD,
                        "date":     latest_date,
                        "message":  f"7-day MAE={latest_mae:.2f}h exceeds CRITICAL threshold ({MAE_ALERT_THRESHOLD}h). Retrain required.",
                    })
                elif latest_mae > MAE_WARN_THRESHOLD:
                    report["alerts"].append({
                        "model":    "model1_waiting_time",
                        "severity": "WARNING",
                        "metric":   "rolling_7d_mae",
                        "value":    latest_mae,
                        "threshold": MAE_WARN_THRESHOLD,
                        "date":     latest_date,
                        "message":  f"7-day MAE={latest_mae:.2f}h exceeds WARNING threshold ({MAE_WARN_THRESHOLD}h). Monitor closely.",
                    })
    else:
        log.warning("Model 1: insufficient records with actuals (%d)", len(m1_df))
        report["model1_waiting_time"] = {"n": len(m1_df), "note": "insufficient_data"}

    # ── Model 3: Congestion risk ──────────────────────────────────────────────
    m3_req  = ["congestion_flag_actual", "congestion_flag_predicted"]
    m3_df   = df.dropna(subset=m3_req) if all(c in df.columns for c in m3_req) else pd.DataFrame()

    if len(m3_df) >= 10:
        log.info("Model 3 (congestion): %d records with actuals", len(m3_df))
        scores = m3_df["congestion_score"].values if "congestion_score" in m3_df.columns else None
        m3_metrics = compute_classification_metrics(
            m3_df["congestion_flag_actual"].values.astype(int),
            m3_df["congestion_flag_predicted"].values.astype(int),
            scores,
        )
        m3_metrics["vs_training"] = {
            "precision_delta": round(m3_metrics["precision"] - TRAIN_PRECISION, 4) if m3_metrics["precision"] else None,
            "recall_delta":    round(m3_metrics["recall"]    - TRAIN_RECALL,    4) if m3_metrics["recall"]    else None,
        }
        report["model3_congestion"] = m3_metrics

        log.info("  Precision=%.3f  Recall=%.3f  F1=%.3f  AUC=%s",
                 m3_metrics["precision"] or 0,
                 m3_metrics["recall"]    or 0,
                 m3_metrics["f1"]        or 0,
                 f"{m3_metrics['auc']:.3f}" if m3_metrics["auc"] else "N/A")

        # Rolling series
        if "ata_actual" in m3_df.columns:
            rolling_c = rolling_congestion_series(m3_df)
            report["rolling_congestion_by_day"] = rolling_c.to_dict(orient="records")

            if len(rolling_c) > 0:
                latest = rolling_c.iloc[-1]
                latest_prec = latest.get("precision")
                latest_rec  = latest.get("recall")
                latest_date = latest.get("date", "unknown")

                if latest_prec is not None and latest_prec < PRECISION_ALERT_THRESHOLD:
                    report["alerts"].append({
                        "model":     "model3_congestion",
                        "severity":  "CRITICAL",
                        "metric":    "rolling_7d_precision",
                        "value":     latest_prec,
                        "threshold": PRECISION_ALERT_THRESHOLD,
                        "date":      latest_date,
                        "message":   (
                            f"7-day precision={latest_prec:.3f} < {PRECISION_ALERT_THRESHOLD}. "
                            "Alert fatigue risk. Retrain or adjust threshold."
                        ),
                    })

                if latest_rec is not None and latest_rec < RECALL_ALERT_THRESHOLD:
                    report["alerts"].append({
                        "model":     "model3_congestion",
                        "severity":  "CRITICAL",
                        "metric":    "rolling_7d_recall",
                        "value":     latest_rec,
                        "threshold": RECALL_ALERT_THRESHOLD,
                        "date":      latest_date,
                        "message":   (
                            f"7-day recall={latest_rec:.3f} < {RECALL_ALERT_THRESHOLD}. "
                            "Missing too many congestion events. Retrain immediately."
                        ),
                    })

                if m3_metrics.get("auc") and m3_metrics["auc"] < AUC_WARN_THRESHOLD:
                    report["alerts"].append({
                        "model":     "model3_congestion",
                        "severity":  "WARNING",
                        "metric":    "auc",
                        "value":     m3_metrics["auc"],
                        "threshold": AUC_WARN_THRESHOLD,
                        "date":      latest_date,
                        "message":   f"AUC={m3_metrics['auc']:.3f} < {AUC_WARN_THRESHOLD}. Model discriminability degraded.",
                    })
    else:
        log.warning("Model 3: insufficient records with actuals (%d)", len(m3_df))
        report["model3_congestion"] = {"n": len(m3_df), "note": "insufficient_data"}

    # ── Model 2: Berth occupancy ──────────────────────────────────────────────
    m2_req = ["occupancy_class_actual", "occupancy_class_predicted"]
    m2_df  = df.dropna(subset=m2_req) if all(c in df.columns for c in m2_req) else pd.DataFrame()

    if len(m2_df) >= 10:
        log.info("Model 2 (occupancy): %d records with actuals", len(m2_df))
        m2_metrics = compute_accuracy(
            m2_df["occupancy_class_actual"].values,
            m2_df["occupancy_class_predicted"].values,
        )
        report["model2_occupancy"] = m2_metrics
        log.info("  Accuracy=%.3f", m2_metrics["accuracy"] or 0)

        if m2_metrics["accuracy"] is not None and m2_metrics["accuracy"] < OCC_ALERT_THRESHOLD:
            report["alerts"].append({
                "model":     "model2_occupancy",
                "severity":  "WARNING",
                "metric":    "accuracy",
                "value":     m2_metrics["accuracy"],
                "threshold": OCC_ALERT_THRESHOLD,
                "message":   (
                    f"Occupancy accuracy={m2_metrics['accuracy']:.3f} < {OCC_ALERT_THRESHOLD}. "
                    "Berth forecast reliability degraded."
                ),
            })
    else:
        log.info("Model 2: no actuals available (%d)", len(m2_df))
        report["model2_occupancy"] = {"n": len(m2_df), "note": "insufficient_data"}

    # ── Summary ───────────────────────────────────────────────────────────────
    n_critical = sum(1 for a in report["alerts"] if a["severity"] == "CRITICAL")
    n_warnings = sum(1 for a in report["alerts"] if a["severity"] == "WARNING")

    report["degradation_detected"] = len(report["alerts"]) > 0
    report["critical_degradation"] = n_critical >= 1

    log.info("")
    log.info("=== PERFORMANCE SUMMARY ===")
    log.info("  Critical alerts: %d", n_critical)
    log.info("  Warnings:        %d", n_warnings)
    log.info("  Degradation:     %s", report["degradation_detected"])
    log.info("  Critical:        %s", report["critical_degradation"])

    if report["critical_degradation"]:
        log.warning("  ACTION REQUIRED: Run retraining pipeline.")
        log.warning("  Command: python retrain.py --input data/port_calls.parquet --validate")
    elif report["degradation_detected"]:
        log.info("  MONITOR: Performance degrading. Schedule retrain review.")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SAMPLE CSV (for testing without a live DB)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_predictions(n: int = 500, output_path: str = "monitoring/sample_predictions.csv"):
    """
    Generate a realistic sample predictions CSV for testing.
    Simulates a slightly degraded model (MAE ~2.5h vs training 1.22h).
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="3h")

    actual_wait = rng.exponential(scale=6.0, size=n).clip(0, 48)
    # Simulate slight degradation: add extra bias + noise
    forecast_wait = np.clip(actual_wait + rng.normal(0, 2.5, n), 0, 72)

    # Congestion: actual = top 20%
    threshold = np.percentile(actual_wait, 80)
    congestion_actual = (actual_wait >= threshold).astype(int)
    congestion_score  = 1 / (1 + np.exp(-((actual_wait - threshold) / 3 + rng.normal(0, 0.5, n))))
    congestion_pred   = (congestion_score >= 0.892).astype(int)

    # Occupancy
    occ_map    = {0: "Low", 1: "Medium", 2: "High"}
    occ_actual = rng.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20])
    occ_pred   = np.where(rng.random(n) < 0.80, occ_actual,
                          rng.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20]))

    df = pd.DataFrame({
        "ata_actual":                  dates,
        "waiting_anchor_hours_actual": actual_wait.round(1),
        "waiting_anchor_forecast":     forecast_wait.round(1),
        "congestion_flag_actual":      congestion_actual,
        "congestion_flag_predicted":   congestion_pred,
        "congestion_score":            congestion_score.round(4),
        "occupancy_class_actual":      [occ_map[v] for v in occ_actual],
        "occupancy_class_predicted":   [occ_map[v] for v in occ_pred],
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Sample predictions written: %s (%d rows)", output_path, n)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model performance degradation monitoring")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--predictions", help="CSV file with predictions + actuals")
    src.add_argument("--db-url",      help="PostgreSQL URL (postgresql://user:pass@host/db)")
    src.add_argument("--generate-sample", action="store_true",
                     help="Generate a sample predictions CSV and run check on it")

    parser.add_argument("--db-days",   type=int, default=90,
                        help="Days of history to load from DB (default: 90)")
    parser.add_argument("--report",    default="monitoring/performance_report.json",
                        help="Output JSON report path")
    parser.add_argument("--exit-on-degradation", action="store_true",
                        help="Exit with code 1 if critical degradation detected")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.generate_sample:
        sample_path = "monitoring/sample_predictions.csv"
        generate_sample_predictions(output_path=sample_path)
        df = load_from_csv(sample_path)

    elif args.predictions:
        log.info("Loading predictions from: %s", args.predictions)
        df = load_from_csv(args.predictions)

    elif args.db_url:
        log.info("Loading predictions from database (last %d days)...", args.db_days)
        df = load_from_db(args.db_url, days=args.db_days)

    else:
        # Default: look for the standard CSV location
        default_csv = "monitoring/predictions_log.csv"
        if Path(default_csv).exists():
            log.info("Loading default predictions log: %s", default_csv)
            df = load_from_csv(default_csv)
        else:
            log.error(
                "No data source specified. Use --predictions, --db-url, or --generate-sample.\n"
                "  Example: python monitoring/model_performance.py --generate-sample"
            )
            sys.exit(1)

    log.info("Loaded %d records from %s to %s",
             len(df),
             df["ata_actual"].min().date() if "ata_actual" in df else "?",
             df["ata_actual"].max().date() if "ata_actual" in df else "?")

    # ── Run checks ────────────────────────────────────────────────────────────
    report = run_performance_check(df)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report saved: %s", report_path)

    if args.exit_on_degradation and report["critical_degradation"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
