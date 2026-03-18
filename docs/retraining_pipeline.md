# Retraining Pipeline — Port Intelligence Platform

**Cadence:** Monthly (1st of each month, 02:00 UTC)
**Trigger:** Scheduled cron + drift alert
**Owner:** Data Engineering team
**Script:** `retrain.py`

---

## 1. Pipeline Overview

```
Day 1 of month, 02:00 UTC
         │
         ▼
[1] Export TOS data          → TOS system exports last N months to CSV
         │
         ▼
[2] Validate & Ingest        → retrain.py: schema check, quality rules
         │
         ▼
[3] Feature Engineering      → features.py: build 46-feature matrix
         │
         ▼
[4] Time-Series Split        → 80/10/10, 28-day gap before val/test
         │
         ▼
[5] Train 3 Models           → XGBoost + LightGBM (same hyperparams)
         │
         ▼
[6] Validate Performance     → Assert MAE<4h, MAPE<25%, R²>0.75, Prec>0.80
         │            │
         │            └─► FAIL: keep old models, send alert email
         ▼
[7] Archive old models       → models/archive/YYYY-MM-DD/
         │
         ▼
[8] Deploy new models        → overwrite models/*.pkl
         │
         ▼
[9] Restart API              → docker-compose restart api
         │
         ▼
[10] Post-deploy smoke test  → curl /health + /predict_vessel
         │
         ▼
[11] Log to DB               → INSERT INTO model_training_log
```

---

## 2. Minimum Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Training period | 6 months | 18 months |
| Total rows | 10,000 | 50,000+ |
| Haifa calls | 4,000 | 20,000 |
| Ashdod calls | 2,500 | 13,000 |
| Completeness (waiting time) | 80% non-null | 95% |
| Date coverage | No gaps > 14 days | No gaps > 7 days |

If minimums are not met, `retrain.py` exits with code 1 and logs a warning.

---

## 3. Running the Retrain Script

```bash
# With real TOS data (CSV export)
python retrain.py \
  --input /data/tos_export_2026-03.csv \
  --output-dir models/ \
  --min-rows 10000 \
  --validate

# With historical parquet (append to existing)
python retrain.py \
  --input data/port_calls.parquet \
  --output-dir models/ \
  --validate

# Dry run (validate + train but don't overwrite models)
python retrain.py --input data/port_calls.parquet --dry-run

# Force retrain even if metrics slightly below target
python retrain.py --input data/port_calls.parquet --force
```

---

## 4. Expected Output

```
models/
├── waiting_time_ensemble.pkl   ← new model (atomic replace)
├── berth_occupancy.pkl
├── congestion_risk.pkl
├── archive/
│   └── 2026-03-01/
│       ├── waiting_time_ensemble.pkl
│       ├── berth_occupancy.pkl
│       └── congestion_risk.pkl
└── model_cards/
    ├── model1_waiting_time.md  ← regenerated with new metrics
    ├── model2_berth_occupancy.md
    └── model3_congestion_risk.md
```

---

## 5. Cron Setup

```bash
# /etc/cron.d/port-retrain
# Run monthly on the 1st at 02:00 UTC
0 2 1 * * portuser cd /opt/port-intelligence && \
    python retrain.py \
      --input /data/tos_latest.csv \
      --output-dir models/ \
      --validate \
      >> /var/log/port-retrain.log 2>&1 && \
    docker-compose restart api
```

---

## 6. Performance Gates

The pipeline will reject new models (keep previous) if:

| Model | Metric | Gate |
|-------|--------|------|
| Waiting Time | MAE | > 6h (50% degradation) |
| Waiting Time | R² | < 0.60 |
| Congestion Risk | Precision @ recall=0.80 | < 0.70 |
| Any model | Test set size | < 500 rows |

---

## 7. Rollback Procedure

```bash
# If new models cause issues, rollback to last archived version
ls models/archive/
# → 2026-02-01/  2026-01-01/  2025-12-01/

cp models/archive/2026-02-01/*.pkl models/
docker-compose restart api

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

---

## 8. Drift-Triggered Retraining

The monitoring system (`monitoring/data_drift.py`) can trigger out-of-schedule retraining:

```
Condition: KS test p-value < 0.05 for ≥ 3 key features
           OR PSI > 0.25 for berth_competition or weather_wind_knots
           OR MAE degradation > 30% over rolling 7-day window

Action: Send alert email → data team reviews → manual approve → retrain
```

See `monitoring/data_drift.py` for configuration.

---

## 9. What Changes Month-to-Month

| What changes | Why |
|-------------|-----|
| Training data window | More recent patterns replace old ones |
| Hyperparameters | Auto-tuned on new data (optional: enable with `--tune`) |
| Congestion threshold | Recalculated as P80 of new training data |
| Decision threshold | Recalculated from new PR curve |
| Model cards | Auto-regenerated with updated metrics |

**What stays the same:**
- Feature engineering logic (`features.py`)
- Database schema
- API contract (endpoints, field names)
- Docker setup

---

## 10. Integration with TOS (Terminal Operating System)

See `docs/real_data_integration.md` for the full TOS field mapping.

Quick summary:
1. TOS exports arrive as daily CSV at `/data/incoming/tos_YYYYMMDD.csv`
2. `retrain.py --append` merges new rows into the training store
3. Full retrain runs on the 1st of each month using the full merged history
