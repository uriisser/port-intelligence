# Data Dictionary — Port Intelligence Platform

**Version:** 1.0 | **Owner:** Data Engineering, Israeli Ports Authority

---

## 1. Core Table: `port_calls_synthetic`

Primary fact table. One row = one vessel port call.

| Column | Type | Unit | Nullable | Description | Business Rule |
|--------|------|------|----------|-------------|---------------|
| `id` | INTEGER | — | No | Surrogate primary key | Auto-increment |
| `port_name` | VARCHAR(10) | — | No | Port identifier | Must be `Haifa` or `Ashdod` |
| `vessel_imo` | VARCHAR(8) | — | Yes | 8-digit IMO number | Format: `9XXXXXXX` (check digit validated) |
| `vessel_name` | VARCHAR(50) | — | Yes | Commercial vessel name | e.g. `Maersk Blue 0012` |
| `vessel_type` | VARCHAR(20) | — | Yes | Cargo type category | One of: `CONTAINER`, `BULK`, `GENERAL_CARGO`, `RORO`, `TANKER` |
| `dwt` | INTEGER | tons | Yes | Deadweight tonnage | Range: 500–500,000 tons |
| `teu_capacity` | INTEGER | TEU | Yes | Nominal TEU capacity | 0 for non-container vessels |
| `loa` | INTEGER | meters | Yes | Length overall | Range: 50–450m (port constraint) |
| `draft` | DECIMAL(4,1) | meters | Yes | Maximum operating draft | Range: 2.0–18.0m |
| `company_name` | VARCHAR(50) | — | Yes | Shipping line operator | e.g. `Maersk`, `MSC`, `CMA CGM` |
| `service_line` | VARCHAR(30) | — | Yes | Trade route name | e.g. `Asia-EU`, `Med-India` |
| `eta_planned` | TIMESTAMP | UTC | No | Planned arrival (as per schedule) | Provided by vessel agent |
| `ata_actual` | TIMESTAMP | UTC | No | Actual arrival at pilot boarding point | Recorded by port operations |
| `atb` | TIMESTAMP | UTC | Yes | Actual time of berthing | When vessel lines are secured |
| `etd` | TIMESTAMP | UTC | Yes | Estimated departure at berthing time | Agreed with terminal |
| `atd_actual` | TIMESTAMP | UTC | No | Actual departure time | Lines off |
| `waiting_anchor_hours` | DECIMAL(5,1) | hours | Auto | **Generated**: `(atb - ata_actual) / 3600` | Primary ML target. 0 = direct berth |
| `waiting_berth_hours` | DECIMAL(5,1) | hours | Auto | **Generated**: `(etd - atb) / 3600` | Productive berth time |
| `berth_id` | VARCHAR(20) | — | Yes | Assigned berth identifier | Haifa: H01–H20, Ashdod: A01–A15 |
| `cranes_used` | INTEGER | count | Yes | Number of cranes assigned | Range: 0–8 |
| `cargo_tons` | DECIMAL(10,0) | tons | Yes | Total cargo handled (load + discharge) | |
| `teu_loaded` | INTEGER | TEU | No | TEU loaded onto vessel | 0 for non-container |
| `teu_discharged` | INTEGER | TEU | No | TEU discharged from vessel | 0 for non-container |
| `weather_wind_knots` | DECIMAL(4,1) | knots | Yes | Wind speed at time of arrival | Source: IMS weather station |
| `berth_competition` | DECIMAL(5,3) | ratio | Yes | Same-day berth demand pressure | Gamma(1.5 × day_mult, 0.7). High = congested |
| `created_date` | TIMESTAMP | UTC | Auto | Row insertion timestamp | Default: `CURRENT_TIMESTAMP` |

### Key Derived Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Waiting Time** | `atb - ata_actual` | Time at anchor. Target P80 < 12h |
| **Berth Productivity** | `(teu_loaded + teu_discharged) / waiting_berth_hours` | TEU/hour |
| **Berth Utilization** | `waiting_berth_hours / (24 × berth_count)` | % of berth capacity used |
| **On-time Arrival** | `\|ata_actual - eta_planned\| < 2h` | Schedule reliability |
| **TEU Imbalance** | `\|teu_loaded - teu_discharged\| / total_teu` | Trade direction signal |
| **Load Factor** | `(teu_loaded + teu_discharged) / teu_capacity` | Vessel fill rate |

---

## 2. Prediction Log: `vessel_predictions`

One row per API call to `POST /predict_vessel`. Actuals backfilled after arrival.

| Column | Type | Description |
|--------|------|-------------|
| `prediction_id` | UUID | Primary key |
| `vessel_id` | VARCHAR(50) | Caller-supplied vessel identifier |
| `port_name` | VARCHAR(10) | Target port |
| `predicted_at` | TIMESTAMP | When prediction was made |
| `eta_planned` | TIMESTAMP | Input ETA |
| `waiting_anchor_forecast` | DECIMAL(5,1) | Model 1 output (hours) |
| `ci_low` / `ci_high` | DECIMAL(5,1) | 90% confidence interval bounds |
| `recommended_berth` | VARCHAR(10) | Model berth recommendation |
| `congestion_risk` | DECIMAL(5,3) | Model 3 probability score 0–1 |
| `congestion_flag` | BOOLEAN | `TRUE` if `congestion_risk ≥ 0.892` (trained threshold) |
| `occupancy_class` | VARCHAR(10) | Model 2 output: `Low`/`Medium`/`High` |
| `model_version` | VARCHAR(30) | Active model version at prediction time |
| `actual_waiting_hours` | DECIMAL(5,1) | Actual wait (backfilled from TOS) |
| `mae_error` | DECIMAL(5,1) | `actual - forecast` (computed on backfill) |

---

## 3. Berth Forecast Store: `hourly_berth_forecast`

One row per berth × date × hour combination.

| Column | Type | Description |
|--------|------|-------------|
| `forecast_id` | SERIAL | Primary key |
| `berth_id` | VARCHAR(10) | Berth identifier |
| `port_name` | VARCHAR(10) | Port |
| `forecast_date` | DATE | Forecast target date |
| `hour` | SMALLINT | Hour 0–23 (UTC) |
| `utilization` | DECIMAL(4,3) | 0.0–1.0 composite score |
| `occupancy_class` | VARCHAR(10) | `Low` / `Medium` / `High` |
| `prob_low/medium/high` | DECIMAL(4,3) | Model 2 class probabilities |
| `created_at` | TIMESTAMP | Forecast generation time |

---

## 4. Feature Dictionary (ML Models)

All 46 features used in training. Grouped by category.

### 4.1 Temporal Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `hour_of_day` | float | 0–24 | Decimal hour of ATA (e.g. 10.5 = 10:30) |
| `day_of_week` | int | 0–6 | Monday=0, Sunday=6 |
| `day_of_month` | int | 1–31 | Calendar day |
| `month` | int | 1–12 | Calendar month |
| `week_of_year` | int | 1–53 | ISO week number |
| `quarter` | int | 1–4 | Calendar quarter |
| `is_weekend` | int | 0/1 | 1 if Saturday or Sunday |
| `is_peak_hour` | int | 0/1 | 1 if 08:00–18:00 |
| `holiday_flag` | int | 0/1 | 1 if Jewish holiday (see holiday list) |
| `days_since_holiday` | int | 0–99 | Days to nearest Jewish holiday (any direction) |
| `eta_deviation_min` | float | −120 to 480 | Minutes: positive = late vs planned |
| `hour_sin` / `hour_cos` | float | −1 to 1 | Cyclical encoding of hour |
| `dow_sin` / `dow_cos` | float | −1 to 1 | Cyclical encoding of day of week |
| `month_sin` / `month_cos` | float | −1 to 1 | Cyclical encoding of month |

### 4.2 Vessel Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `company_tier` | int | 1–3 | Tier 1: Maersk/MSC/CMA/COSCO. Tier 2: Hapag/ONE/ZIM. Tier 3: Others |
| `is_container` | int | 0/1 | 1 if vessel_type = CONTAINER |
| `teu_cap_norm` | float | 0–1 | `teu_capacity / 24,000` |
| `dwt_norm` | float | 0–1 | `dwt / 220,000` |
| `loa_norm` | float | 0–1 | `loa / 400` |
| `load_factor` | float | 0–1 | `(teu_loaded + teu_discharged) / teu_capacity` |
| `vessel_type_enc` | int | 0–4 | Label encoded vessel type |
| `vessel_teu_class_enc` | int | 0–7 | Feeder=0, Panamax=1, Post-Panamax=2, Ultra-Large=3 |

### 4.3 Operational Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `arrivals_6h` | int | 0–50 | Vessels arrived at same port in last 6 hours |
| `arrivals_12h` | int | 0–100 | Vessels arrived at same port in last 12 hours |
| `arrivals_24h` | int | 0–200 | Vessels arrived at same port in last 24 hours |
| `berth_competition_ratio` | float | 0–5 | `arrivals_12h / berth_count` |
| `queue_position` | int | 1–99 | Rank within same-day same-port arrivals |
| `crane_sharing_risk` | float | 0–24 | `cranes_used × berth_competition_ratio` |
| `service_frequency` | float | 0–1 | Normalized frequency of this service line |
| `berth_competition` | float | 0–5 | **Key causal feature** — see generation logic |

### 4.4 Port Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `port_haifa` | int | 0/1 | 1 if Haifa, 0 if Ashdod |
| `berth_num` | int | 1–20 | Numeric part of berth ID |
| `berth_zone_enc` | int | 0/1 | H=0 (Haifa), A=1 (Ashdod) |
| `port_name_enc` | int | 0/1 | Label encoded port |

### 4.5 Weather Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `weather_wind_knots` | float | 0–60 | Wind speed at arrival (knots). Source: IMS |
| `weather_storm_flag` | int | 0/1 | 1 if `weather_wind_knots > 25` |

### 4.6 Cargo Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `cargo_tons_log` | float | 0–15 | `log1p(cargo_tons)` — reduces right skew |
| `teu_imbalance` | float | 0–1 | `|teu_loaded - teu_discharged| / total_teu` |
| `cranes_used` | int | 0–8 | Number of cranes assigned |
| `dwt` | int | 500–500K | Raw DWT (complements normalized version) |

---

## 5. Business Rules

### Waiting Time Categories
| Category | Threshold | Action |
|----------|-----------|--------|
| Fast | 0–2h | Normal — no intervention |
| Moderate | 2–6h | Monitor — notify duty manager |
| Delayed | 6–12h | Alert — consider berth swap |
| Congested | > 12h (P80) | Escalate — port authority notification |
| Crisis | > 36h (P95) | Emergency — fleet coordination |

### Congestion Risk Thresholds (Model 3)
| Score | Flag | Response |
|-------|------|----------|
| 0.0–0.40 | Green | Normal operations |
| 0.40–0.70 | Yellow | Duty manager awareness |
| 0.70–0.892 | Orange | Prepare contingency plan |
| ≥ 0.892 | **Red** | Congestion flag = TRUE, alert triggered |

### Jewish Holidays (arrival reduction −30%)
Rosh Hashana, Yom Kippur, Sukkot (first+last days), Pesach (first+last days),
Shavuot, Independence Day. Full list in `generate_data.py:JEWISH_HOLIDAYS`.

---

## 6. Data Quality Rules (for Real TOS Integration)

| Rule | Check | On Failure |
|------|-------|------------|
| No null timestamps | `eta_planned IS NOT NULL AND ata_actual IS NOT NULL AND atd_actual IS NOT NULL` | Reject row |
| Chronological order | `eta_planned ≤ ata_actual + 72h` and `ata_actual ≤ atb ≤ atd_actual` | Flag, review |
| Valid port | `port_name IN ('Haifa','Ashdod')` | Reject row |
| Positive waiting | `atb ≥ ata_actual` | If negative: set `atb = ata_actual` (direct berth) |
| TEU consistency | `teu_loaded + teu_discharged ≤ teu_capacity * 1.05` | Warn (allowance for measurement) |
| Draft limit | `draft BETWEEN 2.0 AND 18.0` | Clip and warn |
| LOA limit | `loa BETWEEN 50 AND 450` | Clip and warn |
| Duplicate IMO | Same `vessel_imo` + same `ata_actual` (±6h) | Flag for dedup review |
