# Port Intelligence Platform — System Architecture

**Version:** 1.0
**Owner:** Data & Analytics Division, Israeli Ports Authority
**Last Updated:** March 2026

---

## 1. End-to-End System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                        │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  ┌─────────────┐  │
│  │  TOS / Port  │   │  AIS Stream  │   │  Weather API │  │  Synthetic  │  │
│  │  Management  │   │  (vessel     │   │  (IMS / wind)│  │  Dataset    │  │
│  │  System      │   │   position)  │   │              │  │  (Phase 1)  │  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘  └──────┬──────┘  │
└─────────┼─────────────────┼─────────────────┼────────────────┼───────────┘
          │                 │                 │                │
          ▼                 ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                                     │
│                                                                             │
│   retrain.py / ETL scripts                                                  │
│   • Validate schema (port_name, vessel_imo, timestamps)                     │
│   • Enforce data quality rules (draft 2-18m, LOA 50-450m)                  │
│   • Compute derived fields (waiting_anchor_hours, load_factor)              │
│   • Write to: PostgreSQL (port_calls_synthetic table)                       │
│   • Write to: data/port_calls.parquet (ML training store)                  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING  (features.py)                  │
│                                                                             │
│   46 features across 6 categories:                                          │
│   • Temporal   : hour_sin/cos, dow_sin/cos, holiday_flag, week_of_year      │
│   • Vessel     : teu_cap_norm, dwt_norm, company_tier, vessel_teu_class     │
│   • Operational: berth_competition_ratio, queue_position, crane_sharing     │
│   • Port       : port_haifa, berth_num, berth_zone_enc                      │
│   • Weather    : weather_wind_knots, weather_storm_flag                     │
│   • Cargo      : load_factor, teu_imbalance, cargo_tons_log                 │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML MODELS  (Phase 2)                                │
│                                                                             │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
│   │  Model 1             │  │  Model 2             │  │  Model 3         │ │
│   │  Waiting Time        │  │  Berth Occupancy     │  │  Congestion Risk │ │
│   │  Regression          │  │  Multiclass          │  │  Binary          │ │
│   │                      │  │                      │  │                  │ │
│   │  XGBoost + LightGBM  │  │  XGBoost Classifier  │  │  XGBoost + class │ │
│   │  Ensemble (w=0.30)   │  │  (Low/Med/High)      │  │  weighting       │ │
│   │                      │  │                      │  │                  │ │
│   │  MAE:  1.22h         │  │  Accuracy: 1.00      │  │  AUC:  0.991     │ │
│   │  MAPE: 17.7%         │  │  Macro F1: 1.00      │  │  Prec: 0.969     │ │
│   │  R²:   0.938         │  │                      │  │  Rec:  0.802     │ │
│   └──────────────────────┘  └──────────────────────┘  └──────────────────┘ │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER  (Phase 3)                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  FastAPI  (api/main.py)                                             │  │
│   │                                                                     │  │
│   │  POST /predict_vessel      → waiting time + CI + berth + congestion │  │
│   │  GET  /berth_forecast/{id} → 24h hourly utilization                 │  │
│   │  GET  /health              → liveness probe                         │  │
│   │  GET  /metrics             → model performance metadata             │  │
│   └────────────────────────────────┬────────────────────────────────────┘  │
│                                    │                                        │
│   ┌────────────────────┐   ┌───────┴──────────┐   ┌────────────────────┐  │
│   │  Redis (cache)     │   │  PostgreSQL       │   │  Uvicorn workers   │  │
│   │  TTL: 5 min        │◄──│  prediction log   │   │  2 processes       │  │
│   │  256MB LRU         │   │  + historical     │   │  port 8000         │  │
│   └────────────────────┘   └───────────────────┘   └────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                  │
│                                                                             │
│   ┌──────────────────────────┐    ┌──────────────────────────────────────┐  │
│   │  Streamlit Dashboard     │    │  External consumers                  │  │
│   │  demo/streamlit_app.py   │    │                                      │  │
│   │                          │    │  • Port operations system (REST)     │  │
│   │  • Live prediction form  │    │  • Vessel scheduling system (REST)   │  │
│   │  • 24h berth heat map    │    │  • BI dashboards (Tableau/Power BI)  │  │
│   │  • Historical KPI charts │    │  • Email alerts (congestion > 80%)   │  │
│   │  • Model accuracy tab    │    │                                      │  │
│   └──────────────────────────┘    └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITORING LAYER  (Phase 4)                         │
│                                                                             │
│   monitoring/data_drift.py          monitoring/model_performance.py         │
│   • KS test per feature             • Daily MAE vs target                   │
│   • PSI (Population Stability)      • Precision/Recall degradation          │
│   • Alert if drift p < 0.05         • Auto-trigger retraining               │
│                                                                             │
│   retrain.py (monthly cron)                                                 │
│   • Ingest new TOS data → validate → engineer features → retrain → deploy   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Data store | PostgreSQL | 15 | ACID, array ops, window functions |
| Cache | Redis | 7 | Prediction cache, TTL-based invalidation |
| ML training | XGBoost + LightGBM | 2.1 / 4.6 | GBDT ensemble, fast inference |
| Feature eng. | pandas + numpy | 2.x | Vectorized rolling windows |
| SHAP | shap | 0.49 | Explainability for BI team |
| API | FastAPI + Uvicorn | 0.115 | Async, OpenAPI auto-docs |
| Dashboard | Streamlit | 1.40 | Rapid BI prototype |
| Containers | Docker + Compose | 24 / 2.x | Reproducible deployment |
| Monitoring | scipy KS test | 1.13 | Distribution drift detection |

---

## 3. Data Flow

```
TOS Export (CSV/Excel)
     │
     ▼  retrain.py --input real_data.csv
validate_schema()         ← checks 24 required columns
compute_waiting_time()    ← ata_actual → atb gap in hours
build_features()          ← 46 features via features.py
train_models()            ← time-series 80/10/10 split
save_models()             ← models/*.pkl
restart_api()             ← SIGTERM → uvicorn reloads new pkl
```

---

## 4. Security & Operations

| Concern | Implementation |
|---------|----------------|
| Auth | API key header (`X-API-Key`) — add in Phase 5 |
| Secrets | Environment variables via `.env` (never in Git) |
| Data retention | `vessel_predictions` — 2 years rolling delete |
| Backup | `pg_dump` daily to object storage |
| Updates | Rolling restart: `docker-compose up -d --no-deps api` |
| Rollback | `git tag v1.0` + `docker tag port-api:v1.0` |

---

## 5. Scaling Path

```
Phase 3 (Current)          Phase 5 (Future)
─────────────────          ────────────────
Single server               Kubernetes cluster
2 Uvicorn workers           HPA on CPU/RPS
PostgreSQL (single)         Aurora PostgreSQL (multi-AZ)
Redis (single)              Redis Cluster
Streamlit                   React + Grafana dashboards
Manual retrain              Airflow DAG (monthly)
```
