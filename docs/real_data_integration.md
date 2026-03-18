# Real Data Integration Guide
## TOS → Port Intelligence Platform

**Purpose:** Replace the Phase 1 synthetic dataset with real port call data
**Audience:** IT integration team, data engineers
**Time to complete:** 2–5 days for a standard TOS system

---

## 1. Required Data Fields

The platform needs these fields from your TOS. All timestamps must be UTC.

| Platform Field | TOS Typical Name | Type | Notes |
|----------------|-----------------|------|-------|
| `port_name` | `PORT_CODE` / `TERMINAL` | string | Map to `Haifa` or `Ashdod` |
| `vessel_imo` | `IMO_NUMBER` / `VESSEL_IMO` | string | 7-digit IMO (we store 8-char with check) |
| `vessel_name` | `VESSEL_NAME` | string | Commercial name |
| `vessel_type` | `VESSEL_TYPE_CODE` | string | Map using type table below |
| `dwt` | `DWT` / `DEADWEIGHT` | int | Tons |
| `teu_capacity` | `TEU_CAPACITY` / `MAX_TEU` | int | 0 for non-container |
| `loa` | `LENGTH_OVERALL` / `LOA_M` | int | Meters |
| `draft` | `MAX_DRAFT` / `DRAFT_M` | float | Meters |
| `company_name` | `SHIPPING_LINE` / `CARRIER` | string | Shipping company |
| `service_line` | `SERVICE_CODE` / `ROUTE` | string | Trade route |
| `eta_planned` | `ETA` / `PLANNED_ARRIVAL` | datetime | UTC |
| `ata_actual` | `ATA` / `ACTUAL_ARRIVAL` | datetime | UTC — pilot boarding point |
| `atb` | `ATB` / `BERTH_TIME` / `ALONGSIDE` | datetime | UTC — lines secured |
| `etd` | `ETD` / `ESTIMATED_DEPARTURE` | datetime | UTC |
| `atd_actual` | `ATD` / `ACTUAL_DEPARTURE` | datetime | UTC — lines off |
| `berth_id` | `BERTH_CODE` / `QUAY_ID` | string | Map to H01–H20 / A01–A15 |
| `cranes_used` | `CRANE_COUNT` / `QC_COUNT` | int | Quay cranes |
| `cargo_tons` | `TOTAL_CARGO_MT` / `CARGO_WEIGHT` | float | Metric tons |
| `teu_loaded` | `TEU_LOADED` / `LOAD_TEU` | int | Outbound TEU |
| `teu_discharged` | `TEU_DISCHARGED` / `DISCH_TEU` | int | Inbound TEU |

### Derived by Platform (do NOT provide)
- `waiting_anchor_hours` — computed from `atb - ata_actual`
- `waiting_berth_hours` — computed from `etd - atb`

### Optional Enrichment (improves model accuracy)
- `weather_wind_knots` — from IMS weather station at port entrance. If not available, the platform uses a default.
- `berth_competition` — number of vessels at anchor simultaneously. If not available, approximated from `arrivals_12h`.

---

## 2. Vessel Type Mapping

Map your TOS vessel type codes to the platform's 5 categories:

| Platform Type | Common TOS Codes |
|---------------|-----------------|
| `CONTAINER` | `CNT`, `CONT`, `CS`, `CC`, `CV`, `Container Ship` |
| `BULK` | `BLK`, `BULK`, `BC`, `Bulk Carrier`, `OBO`, `Ore Carrier` |
| `GENERAL_CARGO` | `GC`, `GEN`, `MPV`, `Multi Purpose`, `Break Bulk` |
| `RORO` | `RO`, `RORO`, `PCC`, `Car Carrier`, `RoRo/LoLo` |
| `TANKER` | `TK`, `TNK`, `OT`, `Chemical Tanker`, `LPG`, `LNG` |

---

## 3. Berth Mapping

Create a mapping file `config/berth_mapping.csv`:

```csv
tos_berth_code,platform_berth_id,port_name
HAIFA-Q1,H01,Haifa
HAIFA-Q2,H02,Haifa
HAIFA-CARMEL,H03,Haifa
...
ASHDOD-T1,A01,Ashdod
ASHDOD-T2,A02,Ashdod
```

---

## 4. CSV Export Format

The `retrain.py` script accepts CSV with these columns (order does not matter):

```csv
port_name,vessel_imo,vessel_name,vessel_type,dwt,teu_capacity,loa,draft,
company_name,service_line,eta_planned,ata_actual,atb,etd,atd_actual,
berth_id,cranes_used,cargo_tons,teu_loaded,teu_discharged,
weather_wind_knots,berth_competition
```

**Date format:** ISO 8601 — `2025-03-15T10:30:00` (UTC, no timezone suffix)

**Example rows:**
```csv
Haifa,91234567,MSC Diana,CONTAINER,80000,8000,280,12.5,MSC,Asia-EU,
2025-03-15T08:00:00,2025-03-15T10:30:00,2025-03-15T14:00:00,
2025-03-16T06:00:00,2025-03-16T07:15:00,
H07,4,82000,4200,3800,12.0,1.3
```

---

## 5. Step-by-Step Integration

### Step 1: Audit your TOS export
```bash
# Check what columns your TOS export contains
python retrain.py --input your_export.csv --audit-only
```
This prints a field mapping report showing which fields were found, which are missing, and which need manual mapping.

### Step 2: Create field mapping file
```python
# config/tos_field_map.json
{
  "port_name":     "TERMINAL",
  "vessel_imo":    "IMO_NO",
  "ata_actual":    "ATA_UTC",
  "atb":           "ALONGSIDE_UTC",
  "teu_loaded":    "LOAD_TEU",
  "teu_discharged":"DISCH_TEU",
  "vessel_type":   {"col": "VES_TYPE", "map": {"CS": "CONTAINER", "BLK": "BULK"}}
}
```

### Step 3: Run ingestion with mapping
```bash
python retrain.py \
  --input /data/tos_export.csv \
  --field-map config/tos_field_map.json \
  --berth-map config/berth_mapping.csv \
  --output-dir models/ \
  --validate
```

### Step 4: Review data quality report
```
=== DATA QUALITY REPORT ===
Rows loaded:           45,823
Rows passed all checks: 44,109  (96.3%)
Rows rejected:          1,714   (3.7%)

Rejection breakdown:
  Null atb (vessel still at sea):     842 (1.8%)
  Chronological violations:           412 (0.9%)
  Duplicate vessel+arrival (±6h):     460 (1.0%)

Warning counts:
  draft outside 2-18m (clipped):      83
  teu_loaded > teu_capacity:          21

Passed quality rows shape: (44109, 22)
Date range: 2023-01-01 → 2025-12-31
Ports: Haifa=27,241  Ashdod=16,868
```

### Step 5: Validate model performance
```
=== MODEL PERFORMANCE ON REAL DATA ===
Model 1 (Waiting Time):  MAE=2.8h  MAPE=22.1%  R²=0.81  [PASS]
Model 2 (Occupancy):     Accuracy=0.84  F1=0.83      [PASS]
Model 3 (Congestion):    Precision=0.86  Recall=0.81  [PASS]
```

### Step 6: Deploy
```bash
docker-compose restart api
make test
```

---

## 6. Ongoing Data Pipeline

### Option A: Daily file drop (simplest)
```bash
# Cron: daily at 01:00 UTC
0 1 * * * python retrain.py --input /data/incoming/tos_$(date +%Y%m%d).csv --append
```
- TOS drops a file with yesterday's completed port calls
- Script appends new rows to the training store
- Full retrain runs on the 1st of each month

### Option B: Database direct connection
```python
# config/tos_db.yaml
host: tos-db.ports.gov.il
port: 5432
database: tos_production
user: port_intelligence_ro   # read-only service account
query: |
  SELECT * FROM port_calls
  WHERE atd_actual >= CURRENT_DATE - INTERVAL '1 day'
    AND atd_actual < CURRENT_DATE
```
```bash
python retrain.py --source-db config/tos_db.yaml --append
```

### Option C: REST API pull
```bash
python retrain.py \
  --source-api https://tos.ports.gov.il/api/v1/port_calls \
  --api-key $TOS_API_KEY \
  --from-date 2026-02-01 \
  --append
```

---

## 7. First-Time Historical Load

```bash
# Load 2 years of history for initial model training
python retrain.py \
  --input /data/tos_history_2024_2025.csv \
  --field-map config/tos_field_map.json \
  --berth-map config/berth_mapping.csv \
  --output-dir models/ \
  --validate \
  --min-rows 50000

# Expected runtime: 5-15 minutes depending on data volume
# Expected output:
#   data/port_calls.parquet (real data)
#   models/*.pkl (trained on real data)
#   models/model_cards/*.md (updated performance)
```

---

## 8. Known TOS Integration Issues

| Issue | Workaround |
|-------|-----------|
| ATB not recorded (vessel goes direct) | Set `atb = ata_actual`, `waiting_anchor_hours = 0` |
| Timestamps in local time (IST = UTC+2/+3) | Use `--tz Asia/Jerusalem` flag in retrain.py |
| Cancelled calls in TOS | Filter rows where `atd_actual IS NULL AND vessel_status = 'CANCELLED'` |
| Same vessel, multiple berth shifts | Keep only first ATB; sum TEU across all shifts |
| Pilot boarding logged as ATB | Subtract ~45 min from ATB if recorded at pilot station |
| Weather data in Beaufort scale | Multiply by 3.5 to approximate knots |
