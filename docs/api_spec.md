# API Specification — Port Intelligence API v1.0

**Base URL:** `http://<host>:8000`
**Interactive docs:** `http://<host>:8000/docs` (Swagger UI)
**Format:** JSON, UTF-8
**Auth:** None (Phase 3). Add `X-API-Key` header in Phase 5.

---

## Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/metrics` | Model performance metadata |
| POST | `/predict_vessel` | Vessel arrival prediction |
| GET | `/berth_forecast/{berth_id}/{date}` | 24h hourly berth utilization |
| GET | `/ports` | Available ports and berths |
| GET | `/vessel_types` | Supported vessel types |

---

## POST `/predict_vessel`

Primary prediction endpoint. Accepts a vessel arrival request and returns a full forecast.

### Request Body

```json
{
  "vessel_id":          "MSC-DIANA-2024",
  "port_name":          "Haifa",
  "eta_planned":        "2025-12-01T10:00:00",
  "ata_actual":         "2025-12-01T10:30:00",
  "vessel_type":        "CONTAINER",
  "teu_capacity":       8000,
  "dwt":                80000,
  "loa":                250,
  "draft":              12.0,
  "company_name":       "Maersk",
  "service_line":       "Asia-EU",
  "berth_id":           "H01",
  "cranes_used":        3,
  "cargo_tons":         80000,
  "teu_loaded":         4000,
  "teu_discharged":     4000,
  "weather_wind_knots": 8.0,
  "berth_competition":  1.2,
  "arrivals_6h":        5,
  "arrivals_12h":       10,
  "arrivals_24h":       20,
  "queue_position":     5
}
```

### Request Field Reference

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `vessel_id` | string | Yes | — | — | Your internal vessel identifier or IMO |
| `port_name` | string | Yes | — | `Haifa` \| `Ashdod` | Destination port |
| `eta_planned` | ISO datetime | Yes | — | UTC | Planned arrival from vessel schedule |
| `ata_actual` | ISO datetime | No | = eta_planned | UTC | Actual arrival if vessel already at anchor |
| `vessel_type` | string | No | `CONTAINER` | See vessel types | Cargo type |
| `teu_capacity` | int | No | 8000 | 0–24000 | Nominal TEU capacity |
| `dwt` | int | No | 80000 | 500–500000 | Deadweight tonnage |
| `loa` | int | No | 250 | 50–450 | Length overall (meters) |
| `draft` | float | No | 12.0 | 2.0–18.0 | Maximum draft (meters) |
| `company_name` | string | No | `Other` | — | Shipping line |
| `service_line` | string | No | `Asia-EU` | — | Trade route |
| `berth_id` | string | No | `H01` | H01–H20, A01–A15 | Requested berth |
| `cranes_used` | int | No | 3 | 0–8 | Planned crane allocation |
| `cargo_tons` | float | No | 80000 | ≥ 0 | Total cargo weight |
| `teu_loaded` | int | No | 4000 | 0–teu_capacity | TEU being loaded |
| `teu_discharged` | int | No | 4000 | 0–teu_capacity | TEU being discharged |
| `weather_wind_knots` | float | No | 8.0 | 0–60 | Current wind speed (knots) |
| `berth_competition` | float | No | 1.0 | 0–5 | Current anchor queue pressure |
| `arrivals_6h` | int | No | 5 | ≥ 0 | Vessels arrived in last 6h at this port |
| `arrivals_12h` | int | No | 10 | ≥ 0 | Vessels arrived in last 12h |
| `arrivals_24h` | int | No | 20 | ≥ 0 | Vessels arrived in last 24h |
| `queue_position` | int | No | 5 | ≥ 1 | Vessel's position in arrival queue |

### Response Body

```json
{
  "vessel_id":                 "MSC-DIANA-2024",
  "port_name":                 "Haifa",
  "waiting_anchor_forecast":   7.5,
  "confidence_interval":       [5.7, 9.3],
  "recommended_berth":         "H10",
  "congestion_risk":           0.050,
  "congestion_flag":           false,
  "occupancy_class":           "Low",
  "occupancy_probabilities":   {"Low": 1.0, "Medium": 0.0, "High": 0.0},
  "prediction_timestamp":      "2026-03-15T10:16:02Z",
  "model_version":             "phase2-v1"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `waiting_anchor_forecast` | float | Predicted anchor waiting time (hours) |
| `confidence_interval` | [float, float] | [low, high] — approx. 90% CI (±1.5 × MAE) |
| `recommended_berth` | string | Suggested berth assignment |
| `congestion_risk` | float | Congestion probability 0.000–1.000 |
| `congestion_flag` | boolean | `true` if risk ≥ 0.892 (trained threshold) |
| `occupancy_class` | string | `Low` / `Medium` / `High` |
| `occupancy_probabilities` | object | Per-class probabilities from Model 2 |
| `model_version` | string | Active model identifier |

### Error Responses

| Code | Meaning | Example |
|------|---------|---------|
| 422 | Validation error | `port_name` not in `{Haifa, Ashdod}` |
| 422 | Feature engineering failed | Timestamp parse error |
| 500 | Model inference error | Model file missing or corrupt |

---

## GET `/berth_forecast/{berth_id}/{forecast_date}`

Returns 24 hourly utilization predictions for a berth on a given date.

### Path Parameters

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `berth_id` | string | `H01` | Berth identifier |
| `forecast_date` | string | `2025-12-01` | Target date (YYYY-MM-DD) |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `port_name` | string | Yes | `Haifa` or `Ashdod` |

### Example Request

```
GET /berth_forecast/H01/2025-12-01?port_name=Haifa
```

### Response Body

```json
{
  "berth_id":      "H01",
  "forecast_date": "2025-12-01",
  "port_name":     "Haifa",
  "predictions": [
    {
      "hour": 0,
      "utilization": 0.200,
      "occupancy_class": "Low",
      "probabilities": {"Low": 1.0, "Medium": 0.0, "High": 0.0}
    },
    {
      "hour": 8,
      "utilization": 0.650,
      "occupancy_class": "Medium",
      "probabilities": {"Low": 0.1, "Medium": 0.8, "High": 0.1}
    }
  ]
}
```

---

## GET `/health`

```json
{
  "status":        "ok",
  "models_loaded": true,
  "cache_enabled": false,
  "timestamp":     "2026-03-15T10:15:22Z"
}
```

---

## GET `/metrics`

Returns test-set performance of all three models.

```json
{
  "model1_waiting_time": {"mae": 1.22, "mape": 17.7, "r2": 0.938},
  "model2_occupancy":    {"accuracy": 1.0, "macro_f1": 1.0},
  "model3_congestion":   {"auc": 0.991, "precision": 0.969, "recall": 0.802},
  "feature_count": 46
}
```

---

## Usage Examples (curl)

```bash
# Health check
curl http://localhost:8000/health

# Predict vessel (minimal fields)
curl -X POST http://localhost:8000/predict_vessel \
  -H "Content-Type: application/json" \
  -d '{"vessel_id":"V001","port_name":"Haifa","eta_planned":"2025-12-01T10:00:00"}'

# 24h berth forecast
curl "http://localhost:8000/berth_forecast/H05/2025-12-01?port_name=Haifa"

# All ports and berths
curl http://localhost:8000/ports
```

## Usage Examples (Python)

```python
import requests

# Predict vessel
response = requests.post(
    "http://localhost:8000/predict_vessel",
    json={
        "vessel_id": "MSC-DIANA-2024",
        "port_name": "Haifa",
        "eta_planned": "2025-12-01T10:00:00",
        "vessel_type": "CONTAINER",
        "teu_capacity": 8000,
        "berth_competition": 1.5,
        "weather_wind_knots": 12.0,
    }
)
result = response.json()
print(f"Wait: {result['waiting_anchor_forecast']}h")
print(f"Congestion: {result['congestion_risk']:.0%}")
print(f"Berth: {result['recommended_berth']}")
```

---

## Rate Limits & SLAs

| Metric | Target |
|--------|--------|
| P99 response time | < 200ms (cached), < 500ms (uncached) |
| Availability | 99.5% |
| Cache TTL | 5 minutes (Redis) |
| Max request body | 1 MB |
| Concurrent requests | 100 (2 Uvicorn workers) |
