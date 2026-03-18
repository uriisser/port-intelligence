"""
api/main.py — Phase 3: Port Intelligence API
FastAPI application serving Phase 2 ML models.

Endpoints:
  POST /predict_vessel        → waiting time, berth recommendation, congestion risk
  GET  /berth_forecast/{id}/{date} → 24h hourly utilization
  GET  /health                → liveness check
  GET  /metrics               → model performance metadata
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, date
from typing import Optional, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.predictor import predictor, build_inference_features

# ── Redis (optional — graceful fallback if unavailable) ───────────────────────
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _redis = redis.from_url(REDIS_URL, socket_connect_timeout=2)
    _redis.ping()
    CACHE_TTL = 300   # 5 minutes
    USE_CACHE = True
    logging.info("Redis cache: connected")
except Exception:
    _redis = None
    USE_CACHE = False
    logging.warning("Redis unavailable — cache disabled")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

VESSEL_TYPES = {"CONTAINER", "BULK", "GENERAL_CARGO", "RORO", "TANKER"}
PORTS        = {"Haifa", "Ashdod"}

class VesselRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "vessel_id":          "MSC-HAIFA-001",
                "port_name":          "Haifa",
                "eta_planned":        "2026-03-15T14:00:00",
                "vessel_type":        "CONTAINER",
                "teu_capacity":       8000,
                "dwt":                95000,
                "loa":                300,
                "cargo_tons":         45000,
                "berth_competition":  1.5,
                "weather_wind_knots": 12.0,
                "arrivals_12h":       8,
            }
        }
    }

    vessel_id:           str  = Field(..., description="IMO or internal ID")
    port_name:           str  = Field(..., description="Haifa | Ashdod")
    eta_planned:         datetime = Field(..., description="Planned ETA (UTC ISO-8601)")
    ata_actual:          Optional[datetime] = Field(None, description="Actual arrival (defaults to eta_planned)")
    vessel_type:         str  = Field("CONTAINER", description="CONTAINER | BULK | GENERAL_CARGO | RORO | TANKER")
    teu_capacity:        int  = Field(8000, ge=0, le=24000)
    dwt:                 int  = Field(80000, ge=500, le=500000)
    loa:                 int  = Field(250, ge=50, le=450)
    draft:               float = Field(12.0, ge=2.0, le=18.0)
    company_name:        str  = Field("Other")
    service_line:        str  = Field("Asia-EU")
    berth_id:            str  = Field("H01")
    cranes_used:         int  = Field(3, ge=0, le=8)
    cargo_tons:          float = Field(80000.0, ge=0)
    teu_loaded:          int  = Field(4000, ge=0)
    teu_discharged:      int  = Field(4000, ge=0)
    weather_wind_knots:  float = Field(8.0, ge=0.0, le=60.0)
    berth_competition:   float = Field(1.0, ge=0.0, le=5.0,
                                       description="Current anchor queue size ratio (0=empty, 5=severe)")
    arrivals_6h:         int  = Field(5, ge=0)
    arrivals_12h:        int  = Field(10, ge=0)
    arrivals_24h:        int  = Field(20, ge=0)
    queue_position:      int  = Field(5, ge=1)

    @field_validator("port_name")
    @classmethod
    def valid_port(cls, v):
        if v not in PORTS:
            raise ValueError(f"port_name must be one of {PORTS}")
        return v

    @field_validator("vessel_type")
    @classmethod
    def valid_vessel_type(cls, v):
        v = v.upper()
        if v not in VESSEL_TYPES:
            raise ValueError(f"vessel_type must be one of {VESSEL_TYPES}")
        return v


class VesselPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    vessel_id:                str
    port_name:                str
    waiting_anchor_forecast:  float  = Field(..., description="Predicted anchor wait (hours)")
    confidence_interval:      list[float] = Field(..., description="[low, high] 90% CI (hours)")
    recommended_berth:        str
    congestion_risk:          float  = Field(..., description="Congestion probability 0-1")
    congestion_flag:          bool   = Field(..., description="True if congestion_risk > decision threshold")
    occupancy_class:          str    = Field(..., description="Low | Medium | High")
    occupancy_probabilities:  dict[str, float]
    prediction_timestamp:     str
    model_version:            str    = "phase2-v1"


class HourlyForecast(BaseModel):
    berth_id:      str
    forecast_date: str
    port_name:     str
    predictions:   list[dict]


class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    cache_enabled: bool
    timestamp:     str


class MetricsResponse(BaseModel):
    model1_waiting_time: dict
    model2_occupancy:    dict
    model3_congestion:   dict
    feature_count:       int


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Port Intelligence API",
    description="Real-time vessel waiting time forecasting for Israeli ports (Haifa & Ashdod)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    log.info("Loading ML models...")
    predictor.load()
    log.info("Models loaded. API ready.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(prefix: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.md5(blob.encode()).hexdigest()}"


def _cache_get(key: str) -> Optional[dict]:
    if not USE_CACHE:
        return None
    try:
        raw = _redis.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(key: str, value: dict):
    if not USE_CACHE:
        return
    try:
        _redis.setex(key, CACHE_TTL, json.dumps(value, default=str))
    except Exception:
        pass


def _decision_threshold() -> float:
    """Load the trained decision threshold from Model 3 bundle."""
    import joblib
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    m3 = joblib.load(os.path.join(MODEL_DIR, "congestion_risk.pkl"))
    return float(m3['decision_threshold'])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded=predictor._m1 is not None,
        cache_enabled=USE_CACHE,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics():
    import joblib
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    m1 = joblib.load(os.path.join(MODEL_DIR, "waiting_time_ensemble.pkl"))
    m2 = joblib.load(os.path.join(MODEL_DIR, "berth_occupancy.pkl"))
    m3 = joblib.load(os.path.join(MODEL_DIR, "congestion_risk.pkl"))
    return MetricsResponse(
        model1_waiting_time=m1['metrics'],
        model2_occupancy=m2['metrics'],
        model3_congestion=m3['metrics'],
        feature_count=len(m1['features']),
    )


@app.post("/predict_vessel", response_model=VesselPrediction, tags=["Predictions"])
def predict_vessel(req: VesselRequest):
    """
    Main prediction endpoint.

    Accepts a vessel arrival request and returns:
    - Predicted anchorage waiting time with confidence interval
    - Recommended berth assignment
    - Congestion risk score and flag
    - Hourly berth occupancy class
    """
    # Check cache
    cache_key = _cache_key("predict_vessel", req.model_dump())
    cached = _cache_get(cache_key)
    if cached:
        log.info("Cache hit for vessel %s", req.vessel_id)
        return VesselPrediction(**cached)

    ata = req.ata_actual or req.eta_planned

    try:
        X = build_inference_features(
            eta_planned=req.eta_planned,
            ata_actual=ata,
            port_name=req.port_name,
            vessel_type=req.vessel_type,
            teu_capacity=req.teu_capacity,
            dwt=req.dwt,
            loa=req.loa,
            company_name=req.company_name,
            service_line=req.service_line,
            berth_id=req.berth_id,
            cranes_used=req.cranes_used,
            cargo_tons=req.cargo_tons,
            teu_loaded=req.teu_loaded,
            teu_discharged=req.teu_discharged,
            weather_wind_knots=req.weather_wind_knots,
            berth_competition=req.berth_competition,
            arrivals_6h=req.arrivals_6h,
            arrivals_12h=req.arrivals_12h,
            arrivals_24h=req.arrivals_24h,
            queue_position=req.queue_position,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    # Run all 3 models
    wait_h, ci_lo, ci_hi = predictor.predict_waiting_time(X)
    occ                   = predictor.predict_occupancy(X)
    cong_prob             = predictor.predict_congestion(X)
    berth                 = predictor.recommend_berth(
        req.port_name, req.loa, req.draft, req.vessel_type)

    # Decision threshold from trained model
    try:
        thresh = _decision_threshold()
    except Exception:
        thresh = 0.5

    result = VesselPrediction(
        vessel_id=req.vessel_id,
        port_name=req.port_name,
        waiting_anchor_forecast=wait_h,
        confidence_interval=[ci_lo, ci_hi],
        recommended_berth=berth,
        congestion_risk=cong_prob,
        congestion_flag=cong_prob >= thresh,
        occupancy_class=occ['occupancy_class'],
        occupancy_probabilities=occ['probabilities'],
        prediction_timestamp=datetime.utcnow().isoformat() + "Z",
    )

    _cache_set(cache_key, result.model_dump())
    log.info("Predicted vessel %s: wait=%.1fh cong=%.2f", req.vessel_id, wait_h, cong_prob)
    return result


@app.get(
    "/berth_forecast/{berth_id}/{forecast_date}",
    response_model=HourlyForecast,
    tags=["Predictions"],
)
def berth_forecast(
    berth_id: str,
    forecast_date: str,
    port_name: str = Query(..., description="Haifa | Ashdod"),
):
    """
    Returns 24 hourly occupancy predictions for a given berth on a given date.

    Example: GET /berth_forecast/H01/2025-12-01?port_name=Haifa
    """
    try:
        fd = date.fromisoformat(forecast_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="forecast_date must be YYYY-MM-DD")

    if port_name not in PORTS:
        raise HTTPException(status_code=422, detail=f"port_name must be one of {PORTS}")

    cache_key = _cache_key("berth_forecast", {"berth_id": berth_id, "date": forecast_date, "port": port_name})
    cached = _cache_get(cache_key)
    if cached:
        return HourlyForecast(**cached)

    hourly = predictor.predict_hourly_berth(berth_id, fd, port_name)

    result = HourlyForecast(
        berth_id=berth_id,
        forecast_date=forecast_date,
        port_name=port_name,
        predictions=hourly,
    )
    _cache_set(cache_key, result.model_dump())
    return result


@app.get("/ports", tags=["Reference"])
def list_ports():
    """List available ports and their berths."""
    return {
        "Haifa":  {"berths": [f"H{i:02d}" for i in range(1, 21)], "total_berths": 20},
        "Ashdod": {"berths": [f"A{i:02d}" for i in range(1, 16)], "total_berths": 15},
    }


@app.get("/vessel_types", tags=["Reference"])
def list_vessel_types():
    return {"vessel_types": sorted(VESSEL_TYPES)}
