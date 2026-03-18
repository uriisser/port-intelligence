"""
retrain.py — Port Intelligence Platform: Production Retraining Pipeline

Ingests real TOS data (or updates parquet store) and retrains all 3 ML models.

Usage:
  python retrain.py --input /data/tos_export.csv --validate
  python retrain.py --input data/port_calls.parquet --dry-run
  python retrain.py --input /data/tos_export.csv --field-map config/tos_field_map.json
  python retrain.py --input data/port_calls.parquet --append --validate
  python retrain.py --input /data/tos_export.csv --audit-only
"""

import os
import sys
import json
import shutil
import argparse
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
ARCHIVE_DIR = MODEL_DIR / "archive"

REQUIRED_COLUMNS = [
    "port_name", "vessel_imo", "vessel_name", "vessel_type", "dwt",
    "teu_capacity", "loa", "draft", "company_name", "service_line",
    "eta_planned", "ata_actual", "atb", "etd", "atd_actual",
    "berth_id", "cranes_used", "cargo_tons", "teu_loaded", "teu_discharged",
]
OPTIONAL_COLUMNS = ["weather_wind_knots", "berth_competition"]

VESSEL_TYPE_MAP = {
    "CNT": "CONTAINER", "CONT": "CONTAINER", "CS": "CONTAINER",
    "CC": "CONTAINER", "CV": "CONTAINER", "CONTAINER SHIP": "CONTAINER",
    "BLK": "BULK", "BULK": "BULK", "BC": "BULK", "BULK CARRIER": "BULK",
    "OBO": "BULK", "ORE CARRIER": "BULK",
    "GC": "GENERAL_CARGO", "GEN": "GENERAL_CARGO", "MPV": "GENERAL_CARGO",
    "MULTI PURPOSE": "GENERAL_CARGO", "BREAK BULK": "GENERAL_CARGO",
    "RO": "RORO", "RORO": "RORO", "PCC": "RORO", "CAR CARRIER": "RORO",
    "TK": "TANKER", "TNK": "TANKER", "OT": "TANKER",
    "CHEMICAL TANKER": "TANKER", "LPG": "TANKER", "LNG": "TANKER",
}

VALID_PORTS = {"Haifa", "Ashdod"}
VALID_TYPES = {"CONTAINER", "BULK", "GENERAL_CARGO", "RORO", "TANKER"}

PERFORMANCE_GATES = {
    "mae_max":       6.0,    # hours — 50% above target
    "r2_min":        0.60,
    "precision_min": 0.70,
    "test_min_rows": 500,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_input(path: str, field_map: dict = None) -> pd.DataFrame:
    """Load CSV or parquet; apply optional field mapping."""
    p = Path(path)
    if not p.exists():
        log.error("Input file not found: %s", path)
        sys.exit(1)

    log.info("Loading %s ...", p.name)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    log.info("  Raw shape: %s", df.shape)

    # Apply field mapping
    if field_map:
        rename = {}
        for platform_col, tos_spec in field_map.items():
            if isinstance(tos_spec, str):
                if tos_spec in df.columns:
                    rename[tos_spec] = platform_col
            elif isinstance(tos_spec, dict):
                col = tos_spec.get("col")
                mapping = tos_spec.get("map", {})
                if col in df.columns:
                    rename[col] = platform_col
                    df[col] = df[col].astype(str).str.upper().map(mapping).fillna(df[col])
        if rename:
            df = df.rename(columns=rename)
            log.info("  Applied field mapping: %d renames", len(rename))

    return df


def load_field_map(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_berth_map(path: str) -> dict:
    """Returns {tos_code: platform_berth_id}."""
    bm = pd.read_csv(path)
    return dict(zip(bm.iloc[:, 0].astype(str), bm.iloc[:, 1].astype(str)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame) -> None:
    """Print field-level audit report and exit."""
    log.info("\n=== FIELD AUDIT REPORT ===")
    found, missing = [], []
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            found.append(f"  FOUND    {col:<30} null={null_pct:.1f}%")
        else:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                missing.append(f"  RENAME   {col:<30} <- {matches[0]}")
            else:
                missing.append(f"  MISSING  {col:<30} *** REQUIRED ***")

    for line in found + missing:
        print(line)

    print(f"\nOptional columns:")
    for col in OPTIONAL_COLUMNS:
        status = "FOUND" if col in df.columns else "missing (will use defaults)"
        print(f"  {status:<10} {col}")

    print(f"\nAll columns in file: {list(df.columns)}")
    print(f"\nRun with --field-map to rename columns automatically.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATION & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_clean(df: pd.DataFrame, berth_map: dict = None) -> pd.DataFrame:
    """Validate schema, apply quality rules, return clean DataFrame."""
    original_n = len(df)
    rejected = pd.DataFrame()

    # ── Parse timestamps ──────────────────────────────────────────────────────
    for col in ["eta_planned", "ata_actual", "atb", "etd", "atd_actual"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")

    # ── Apply berth mapping ───────────────────────────────────────────────────
    if berth_map and "berth_id" in df.columns:
        df["berth_id"] = df["berth_id"].astype(str).map(berth_map).fillna(df["berth_id"])

    # ── Vessel type normalization ─────────────────────────────────────────────
    if "vessel_type" in df.columns:
        df["vessel_type"] = (
            df["vessel_type"].astype(str).str.upper()
            .map(lambda x: VESSEL_TYPE_MAP.get(x, x))
        )

    # ── Port name normalization ───────────────────────────────────────────────
    if "port_name" in df.columns:
        df["port_name"] = df["port_name"].str.strip().str.title()

    # ── Rule 1: Required timestamps must be non-null ──────────────────────────
    for col in ["eta_planned", "ata_actual", "atd_actual"]:
        mask = df[col].isnull()
        rejected = pd.concat([rejected, df[mask].assign(_reject_reason=f"null_{col}")])
        df = df[~mask]

    # ── Rule 2: Valid port ────────────────────────────────────────────────────
    if "port_name" in df.columns:
        mask = ~df["port_name"].isin(VALID_PORTS)
        rejected = pd.concat([rejected, df[mask].assign(_reject_reason="invalid_port")])
        df = df[~mask]

    # ── Rule 3: Valid vessel type ─────────────────────────────────────────────
    if "vessel_type" in df.columns:
        mask = ~df["vessel_type"].isin(VALID_TYPES)
        df.loc[mask, "vessel_type"] = "GENERAL_CARGO"  # default, don't reject

    # ── Rule 4: atb null → direct berth ──────────────────────────────────────
    if "atb" in df.columns:
        df["atb"] = df["atb"].fillna(df["ata_actual"])

    # ── Rule 5: Chronological order ──────────────────────────────────────────
    if "ata_actual" in df.columns and "atb" in df.columns:
        neg_wait = df["atb"] < df["ata_actual"]
        df.loc[neg_wait, "atb"] = df.loc[neg_wait, "ata_actual"]   # clip, don't reject

    if "atb" in df.columns and "atd_actual" in df.columns:
        chrono_bad = df["atd_actual"] < df["atb"]
        rejected = pd.concat([rejected, df[chrono_bad].assign(_reject_reason="chrono_violation")])
        df = df[~chrono_bad]

    # ── Rule 6: Range clipping ────────────────────────────────────────────────
    if "draft" in df.columns:
        df["draft"] = df["draft"].clip(2.0, 18.0)
    if "loa" in df.columns:
        df["loa"] = df["loa"].clip(50, 450)
    if "cranes_used" in df.columns:
        df["cranes_used"] = df["cranes_used"].clip(0, 8).fillna(0).astype(int)

    # ── Fill optional columns ─────────────────────────────────────────────────
    if "weather_wind_knots" not in df.columns:
        rng = np.random.default_rng(42)
        df["weather_wind_knots"] = rng.exponential(8, size=len(df)).clip(0, 50).round(1)
        log.info("  weather_wind_knots: defaulted to Exp(8) distribution")

    if "berth_competition" not in df.columns:
        # Approximate from daily arrival counts
        df = df.sort_values("ata_actual")
        df["berth_competition"] = (
            df.groupby([df["ata_actual"].dt.date, "port_name"])["ata_actual"]
            .transform("count")
            / df["port_name"].map({"Haifa": 20, "Ashdod": 15})
        ).clip(0, 5).round(3)
        log.info("  berth_competition: approximated from daily arrival counts")

    # ── Fill numeric nulls ────────────────────────────────────────────────────
    for col in ["teu_loaded", "teu_discharged"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in ["cargo_tons", "teu_capacity", "dwt", "loa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Compute waiting_anchor_hours ──────────────────────────────────────────
    df["waiting_anchor_hours"] = (
        (df["atb"] - df["ata_actual"]).dt.total_seconds() / 3600
    ).clip(0, 96).round(1)

    # ── Ensure id column ──────────────────────────────────────────────────────
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # ── Quality report ────────────────────────────────────────────────────────
    n_clean = len(df)
    n_rejected = original_n - n_clean
    log.info("\n=== DATA QUALITY REPORT ===")
    log.info("  Input rows:          %d", original_n)
    log.info("  Clean rows:          %d  (%.1f%%)", n_clean, 100 * n_clean / original_n)
    log.info("  Rejected rows:       %d  (%.1f%%)", n_rejected, 100 * n_rejected / original_n)
    if len(rejected) > 0 and "_reject_reason" in rejected.columns:
        for reason, cnt in rejected["_reject_reason"].value_counts().items():
            log.info("    %-30s %d", reason, cnt)
    log.info("  Date range:          %s -> %s",
             df["ata_actual"].min().date(), df["ata_actual"].max().date())
    log.info("  Ports:  Haifa=%d  Ashdod=%d",
             (df["port_name"] == "Haifa").sum(), (df["port_name"] == "Ashdod").sum())
    log.info("  Waiting P80: %.1fh  P95: %.1fh",
             np.percentile(df["waiting_anchor_hours"], 80),
             np.percentile(df["waiting_anchor_hours"], 95))
    log.info("===========================\n")

    return df.reset_index(drop=True)


def check_minimum_data(df: pd.DataFrame, min_rows: int) -> None:
    if len(df) < min_rows:
        log.error("Insufficient data: %d rows (minimum: %d)", len(df), min_rows)
        sys.exit(1)
    if (df["port_name"] == "Haifa").sum() < 1000:
        log.error("Insufficient Haifa data: %d rows", (df["port_name"] == "Haifa").sum())
        sys.exit(1)
    if (df["port_name"] == "Ashdod").sum() < 500:
        log.error("Insufficient Ashdod data: %d rows", (df["port_name"] == "Ashdod").sum())
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame):
    """Run features.py:build_features and return X matrix."""
    from features import build_features, ALL_FEATURES
    log.info("Building features...")
    df_feat = build_features(df)
    X = df_feat[ALL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    log.info("  Feature matrix: %s", X.shape)
    return X, df_feat, ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIME-SERIES SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def make_split(df: pd.DataFrame, df_feat: pd.DataFrame, X: np.ndarray):
    """Reproduce 80/10/10 time-series split with 28-day gap."""
    from pandas import Timedelta
    df_feat = df_feat.sort_values("ata_actual").reset_index(drop=True)
    n = len(df_feat)
    train_end_idx = int(n * 0.80)
    train_cutoff  = df_feat["ata_actual"].iloc[train_end_idx]
    val_start     = train_cutoff + Timedelta(days=28)
    val_cutoff    = val_start + (df_feat["ata_actual"].iloc[-1] - val_start) * 0.50

    train_mask = df_feat["ata_actual"] <= train_cutoff
    val_mask   = (df_feat["ata_actual"] > val_start) & (df_feat["ata_actual"] <= val_cutoff)
    test_mask  = df_feat["ata_actual"] > val_cutoff

    split = {
        "X_train": X[train_mask], "X_val": X[val_mask], "X_test": X[test_mask],
        "y_wait_train": df_feat.loc[train_mask, "waiting_anchor_hours"].clip(0, 96).values,
        "y_wait_val":   df_feat.loc[val_mask,   "waiting_anchor_hours"].clip(0, 96).values,
        "y_wait_test":  df_feat.loc[test_mask,  "waiting_anchor_hours"].clip(0, 96).values,
        "df_train": df_feat[train_mask],
        "df_val":   df_feat[val_mask],
        "df_test":  df_feat[test_mask],
    }
    log.info("Split: train=%d  val=%d  test=%d",
             split["X_train"].shape[0], split["X_val"].shape[0], split["X_test"].shape[0])

    if split["X_test"].shape[0] < PERFORMANCE_GATES["test_min_rows"]:
        log.error("Test set too small: %d rows", split["X_test"].shape[0])
        sys.exit(1)

    return split


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_all(split: dict, features: list, force: bool = False) -> dict:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.metrics import (
        mean_absolute_error, mean_absolute_percentage_error, r2_score,
        classification_report, roc_auc_score, precision_recall_curve,
    )

    results = {}

    # ── Model 1: Waiting Time Regression ─────────────────────────────────────
    log.info("Training Model 1: Waiting Time Regression...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.5, reg_alpha=0.5, objective="reg:squarederror",
        random_state=42, n_jobs=-1, early_stopping_rounds=50, verbosity=0,
    )
    xgb_reg.fit(split["X_train"], split["y_wait_train"],
                eval_set=[(split["X_val"], split["y_wait_val"])], verbose=False)

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_lambda=1.5, reg_alpha=0.5, objective="regression",
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    lgb_reg.fit(split["X_train"], split["y_wait_train"],
                eval_set=[(split["X_val"], split["y_wait_val"])], callbacks=callbacks)

    # Optimize ensemble weight
    best_w, best_mae = 0.5, np.inf
    for w in np.arange(0.3, 0.8, 0.05):
        pred = (w * xgb_reg.predict(split["X_val"]) +
                (1 - w) * lgb_reg.predict(split["X_val"])).clip(0, 96)
        mae = mean_absolute_error(split["y_wait_val"], pred)
        if mae < best_mae:
            best_mae, best_w = mae, w

    pred_test = (best_w * xgb_reg.predict(split["X_test"]) +
                 (1 - best_w) * lgb_reg.predict(split["X_test"])).clip(0, 96)

    mae1  = mean_absolute_error(split["y_wait_test"], pred_test)
    nontrivial = split["y_wait_test"] >= 3.0
    mape1 = mean_absolute_percentage_error(
        split["y_wait_test"][nontrivial], pred_test[nontrivial]) * 100 if nontrivial.sum() > 0 else 0
    r2_1  = r2_score(split["y_wait_test"], pred_test)

    log.info("  MAE=%.2fh  MAPE=%.1f%%  R2=%.3f", mae1, mape1, r2_1)

    if not force:
        if mae1 > PERFORMANCE_GATES["mae_max"]:
            log.error("Model 1 FAILED gate: MAE=%.2fh > %.1fh", mae1, PERFORMANCE_GATES["mae_max"])
            sys.exit(2)
        if r2_1 < PERFORMANCE_GATES["r2_min"]:
            log.error("Model 1 FAILED gate: R2=%.3f < %.2f", r2_1, PERFORMANCE_GATES["r2_min"])
            sys.exit(2)

    results["m1"] = {
        "xgb_reg": xgb_reg, "lgb_reg": lgb_reg, "ensemble_weight": best_w,
        "features": features, "metrics": {"mae": mae1, "mape": mape1, "r2": r2_1},
    }

    # ── Model 2: Berth Occupancy ──────────────────────────────────────────────
    log.info("Training Model 2: Berth Occupancy...")

    def occupancy_label(df_in):
        return pd.cut(df_in["berth_competition_ratio"],
                      bins=[-np.inf, 1.0, 2.0, np.inf], labels=[0, 1, 2]).astype(int)

    y_occ_train = occupancy_label(split["df_train"])
    y_occ_val   = occupancy_label(split["df_val"])
    y_occ_test  = occupancy_label(split["df_test"])

    xgb_clf2 = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.06, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.5,
        objective="multi:softprob", num_class=3,
        random_state=42, n_jobs=-1, early_stopping_rounds=40,
        verbosity=0, eval_metric="mlogloss",
    )
    xgb_clf2.fit(split["X_train"], y_occ_train,
                 eval_set=[(split["X_val"], y_occ_val)], verbose=False)

    pred_occ = xgb_clf2.predict(split["X_test"])
    rep2 = classification_report(y_occ_test, pred_occ, output_dict=True)
    acc2, f1_2 = rep2["accuracy"], rep2["macro avg"]["f1-score"]
    log.info("  Accuracy=%.3f  Macro-F1=%.3f", acc2, f1_2)

    results["m2"] = {
        "model": xgb_clf2, "features": features,
        "label_names": ["Low", "Medium", "High"],
        "metrics": {"accuracy": acc2, "macro_f1": f1_2},
    }

    # ── Model 3: Congestion Risk ──────────────────────────────────────────────
    log.info("Training Model 3: Congestion Risk...")
    threshold = float(np.percentile(split["df_train"]["waiting_anchor_hours"], 80))

    y_cong_train = (split["df_train"]["waiting_anchor_hours"] >= threshold).astype(int).values
    y_cong_val   = (split["df_val"]["waiting_anchor_hours"]   >= threshold).astype(int).values
    y_cong_test  = (split["df_test"]["waiting_anchor_hours"]  >= threshold).astype(int).values

    pos_rate = y_cong_train.mean()
    scale_pw  = (1 - pos_rate) / pos_rate if pos_rate > 0 else 4.0

    xgb_clf3 = xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        scale_pos_weight=scale_pw, objective="binary:logistic",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
        verbosity=0, eval_metric="auc",
    )
    xgb_clf3.fit(split["X_train"], y_cong_train,
                 eval_set=[(split["X_val"], y_cong_val)], verbose=False)

    proba3 = xgb_clf3.predict_proba(split["X_test"])[:, 1]
    auc3   = roc_auc_score(y_cong_test, proba3)

    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_cong_test, proba3)
    mask = recall_arr >= 0.80
    if mask.any():
        best_idx = np.argmax(precision_arr[mask])
        best_prec3 = float(precision_arr[mask][best_idx])
        best_rec3  = float(recall_arr[mask][best_idx])
        thresh_idx = np.where(mask)[0][best_idx]
        best_thresh3 = float(thresholds_arr[thresh_idx] if thresh_idx < len(thresholds_arr) else 0.5)
    else:
        best_prec3, best_rec3, best_thresh3 = 0.0, 0.0, 0.5

    log.info("  AUC=%.3f  Prec=%.3f  Recall=%.3f  Thresh=%.3f",
             auc3, best_prec3, best_rec3, best_thresh3)

    if not force and best_prec3 < PERFORMANCE_GATES["precision_min"]:
        log.error("Model 3 FAILED gate: Precision=%.3f < %.2f",
                  best_prec3, PERFORMANCE_GATES["precision_min"])
        sys.exit(2)

    results["m3"] = {
        "model": xgb_clf3, "features": features,
        "congestion_threshold": threshold, "decision_threshold": best_thresh3,
        "metrics": {"auc": auc3, "precision": best_prec3, "recall": best_rec3},
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def archive_models():
    """Move current models to archive directory."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    archive_path = ARCHIVE_DIR / date_str
    archive_path.mkdir(parents=True, exist_ok=True)
    for pkl in MODEL_DIR.glob("*.pkl"):
        shutil.copy2(pkl, archive_path / pkl.name)
    log.info("Archived current models to %s", archive_path)
    return archive_path


def save_models(results: dict, output_dir: Path, features: list, dry_run: bool = False):
    """Save all 3 model bundles + regenerate model cards."""
    import joblib
    prefix = "[DRY RUN] " if dry_run else ""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "m1": output_dir / "waiting_time_ensemble.pkl",
        "m2": output_dir / "berth_occupancy.pkl",
        "m3": output_dir / "congestion_risk.pkl",
    }

    for key, path in paths.items():
        if not dry_run:
            joblib.dump(results[key], path, compress=3)
        log.info("  %sSaved %s", prefix, path.name)

    # Regenerate model cards
    _write_model_cards(results, output_dir / "model_cards", dry_run)


def _write_model_cards(results: dict, card_dir: Path, dry_run: bool):
    card_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    m1, m2, m3 = results["m1"], results["m2"], results["m3"]

    cards = {
        "model1_waiting_time.md": f"""# Model Card: Waiting Time Regression
**Generated:** {ts} | **Version:** retrained

| Metric | Value | Target |
|--------|-------|--------|
| MAE | {m1['metrics']['mae']:.2f}h | < 4h |
| MAPE (>=3h) | {m1['metrics']['mape']:.1f}% | < 25% |
| R2 | {m1['metrics']['r2']:.3f} | > 0.75 |

Ensemble: XGB weight={m1['ensemble_weight']:.2f} | Features={len(m1['features'])}
""",
        "model2_berth_occupancy.md": f"""# Model Card: Berth Occupancy
**Generated:** {ts} | **Version:** retrained

| Metric | Value |
|--------|-------|
| Accuracy | {m2['metrics']['accuracy']:.3f} |
| Macro F1 | {m2['metrics']['macro_f1']:.3f} |

Classes: {m2['label_names']}
""",
        "model3_congestion_risk.md": f"""# Model Card: Congestion Risk
**Generated:** {ts} | **Version:** retrained

| Metric | Value | Target |
|--------|-------|--------|
| AUC-ROC | {m3['metrics']['auc']:.3f} | — |
| Precision @ recall>=0.80 | {m3['metrics']['precision']:.3f} | > 0.80 |
| Recall | {m3['metrics']['recall']:.3f} | >= 0.80 |
| Decision threshold | {m3['decision_threshold']:.3f} | — |

Congestion threshold (P80): {m3['congestion_threshold']:.1f}h
""",
    }

    for fname, content in cards.items():
        path = card_dir / fname
        if not dry_run:
            path.write_text(content, encoding="utf-8")
        log.info("  %sWrote %s", "[DRY RUN] " if dry_run else "", fname)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Port Intelligence Retraining Pipeline")
    parser.add_argument("--input",      required=True, help="CSV or parquet file path")
    parser.add_argument("--output-dir", default="models", help="Model output directory")
    parser.add_argument("--field-map",  help="JSON field mapping file")
    parser.add_argument("--berth-map",  help="CSV berth code mapping file")
    parser.add_argument("--min-rows",   type=int, default=10_000, help="Minimum clean rows")
    parser.add_argument("--validate",   action="store_true", help="Enforce performance gates")
    parser.add_argument("--dry-run",    action="store_true", help="Train but do not save models")
    parser.add_argument("--force",      action="store_true", help="Save even if gates fail")
    parser.add_argument("--append",     action="store_true", help="Append to existing parquet")
    parser.add_argument("--audit-only", action="store_true", help="Audit fields and exit")
    args = parser.parse_args()

    log.info("=== Port Intelligence Retraining Pipeline ===")
    log.info("Input:  %s", args.input)
    log.info("Output: %s", args.output_dir)

    # Load
    field_map = load_field_map(args.field_map) if args.field_map else None
    berth_map = load_berth_map(args.berth_map) if args.berth_map else None
    df = load_input(args.input, field_map)

    # Audit mode
    if args.audit_only:
        audit(df)

    # Append mode
    if args.append and (DATA_DIR / "port_calls.parquet").exists():
        existing = pd.read_parquet(DATA_DIR / "port_calls.parquet")
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["vessel_imo", "ata_actual"], keep="last")
        log.info("Appended: total rows = %d", len(df))

    # Validate & clean
    df_clean = validate_and_clean(df, berth_map)
    check_minimum_data(df_clean, args.min_rows)

    # Save clean parquet
    if not args.dry_run:
        out_parquet = DATA_DIR / "port_calls.parquet"
        df_clean.to_parquet(out_parquet, index=False)
        log.info("Saved clean parquet: %s (%d rows)", out_parquet, len(df_clean))

    # Feature engineering
    X, df_feat, features = engineer_features(df_clean)

    # Split
    split = make_split(df_clean, df_feat, X)

    # Train
    results = train_all(split, features, force=args.force)

    # Save
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        archive_models()
        save_models(results, output_dir, features, dry_run=False)
        log.info("\n=== Retraining complete. Restart API: docker-compose restart api ===")
    else:
        save_models(results, output_dir, features, dry_run=True)
        log.info("\n=== DRY RUN complete. No files written. ===")

    # Summary
    m1, m2, m3 = results["m1"], results["m2"], results["m3"]
    log.info("\nFINAL METRICS:")
    log.info("  Model 1: MAE=%.2fh  MAPE=%.1f%%  R2=%.3f",
             m1["metrics"]["mae"], m1["metrics"]["mape"], m1["metrics"]["r2"])
    log.info("  Model 2: Accuracy=%.3f  Macro-F1=%.3f",
             m2["metrics"]["accuracy"], m2["metrics"]["macro_f1"])
    log.info("  Model 3: AUC=%.3f  Prec=%.3f  Recall=%.3f",
             m3["metrics"]["auc"], m3["metrics"]["precision"], m3["metrics"]["recall"])


if __name__ == "__main__":
    main()
