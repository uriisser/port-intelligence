"""
predictor.py — Loads Phase 2 models and produces inference feature vectors.

Handles the gap between training features (built from full historical DataFrame)
and inference features (built from a single incoming vessel request).
"""

import os
import math
import numpy as np
import joblib
from datetime import datetime, date
from typing import Optional

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # d:/Code/Shipping
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Jewish holidays 2024-2026 (same set as features.py) ──────────────────────
_HOLIDAY_DATES = {
    "2024-04-22","2024-04-23","2024-04-28","2024-04-29","2024-05-14","2024-06-12",
    "2024-10-02","2024-10-03","2024-10-11","2024-10-12","2024-10-16","2024-10-17",
    "2024-10-23","2024-10-24",
    "2025-04-12","2025-04-13","2025-04-18","2025-04-19","2025-05-01",
    "2025-06-01","2025-06-02","2025-09-22","2025-09-23","2025-10-01","2025-10-02",
    "2025-10-06","2025-10-07","2025-10-13","2025-10-14",
    "2026-03-31","2026-04-01","2026-04-06","2026-04-07","2026-04-23",
    "2026-06-21","2026-09-11","2026-09-12","2026-09-20","2026-09-21",
    "2026-09-25","2026-09-26","2026-10-02","2026-10-03",
}
HOLIDAY_SET = {datetime.strptime(d, "%Y-%m-%d").date() for d in _HOLIDAY_DATES}

COMPANY_TIER = {
    'Maersk': 1, 'MSC': 1, 'CMA CGM': 1, 'COSCO': 1,
    'Hapag-Lloyd': 2, 'ONE': 2, 'Evergreen': 2, 'ZIM': 2,
    'Yang Ming': 2, 'HMM': 2, 'PIL': 3, 'Sinokor': 3,
    'Gold Star': 3, 'Grimaldi': 3, 'OOCL': 2, 'Other': 3,
}

VESSEL_TYPE_ENC = {'CONTAINER': 0, 'BULK': 1, 'GENERAL_CARGO': 2, 'RORO': 3, 'TANKER': 4}
VESSEL_CLASS_ENC = {'Feeder': 0, 'Panamax': 1, 'Post-Panamax': 2, 'Ultra-Large': 3,
                    'BULK': 4, 'GENERAL_CARGO': 5, 'RORO': 6, 'TANKER': 7}
PORT_ENC = {'Haifa': 0, 'Ashdod': 1}
BERTH_ZONE_ENC = {'H': 0, 'A': 1}
SERVICE_LINE_ENC = {
    'Asia-EU': 0, 'Med-India': 1, 'Asia-Med': 2, 'Intra-Med': 3,
    'Asia-US-East': 4, 'Red-Sea-Med': 5, 'Black-Sea-Med': 6, 'North-EU': 7,
    'West-Africa': 8, 'East-Africa': 9, 'Americas': 10, 'Adriatic': 11,
}
BERTH_COUNTS = {'Haifa': 20, 'Ashdod': 15}

# Feature list MUST match training order in features.py:ALL_FEATURES
FEATURE_NAMES = [
    'hour_of_day', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'quarter', 'is_weekend', 'is_peak_hour', 'holiday_flag',
    'days_since_holiday', 'eta_deviation_min',
    'company_tier', 'is_container', 'teu_cap_norm', 'dwt_norm', 'loa_norm',
    'load_factor', 'port_haifa', 'berth_num',
    'arrivals_6h', 'arrivals_12h', 'arrivals_24h',
    'berth_competition_ratio', 'queue_position', 'crane_sharing_risk',
    'service_frequency', 'weather_wind_knots', 'weather_storm_flag',
    'cargo_tons_log', 'teu_imbalance',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'cranes_used', 'dwt',
    'weather_wind_knots', 'weather_storm_flag', 'berth_competition',
    # encoded categoricals
    'vessel_type_enc', 'vessel_teu_class_enc', 'port_name_enc',
    'berth_zone_enc', 'service_line_enc',
]

# ─────────────────────────────────────────────────────────────────────────────

def _teu_class_enc(vessel_type: str, teu: int) -> int:
    if vessel_type != 'CONTAINER':
        return VESSEL_CLASS_ENC.get(vessel_type, 5)
    if teu < 2_000:  return 0  # Feeder
    if teu < 8_000:  return 1  # Panamax
    if teu < 14_000: return 2  # Post-Panamax
    return 3  # Ultra-Large

def _days_to_holiday(d: date) -> int:
    diffs = [abs((datetime.strptime(h, "%Y-%m-%d").date() - d).days) for h in _HOLIDAY_DATES]
    return min(diffs) if diffs else 99


def build_inference_features(
    eta_planned: datetime,
    ata_actual: datetime,
    port_name: str,
    vessel_type: str,
    teu_capacity: int,
    dwt: int,
    loa: int,
    company_name: str,
    service_line: str,
    berth_id: str,
    cranes_used: int,
    cargo_tons: float,
    teu_loaded: int,
    teu_discharged: int,
    weather_wind_knots: float,
    berth_competition: float,
    # Context signals (optional, defaulted)
    arrivals_6h: int = 5,
    arrivals_12h: int = 10,
    arrivals_24h: int = 20,
    queue_position: int = 5,
) -> np.ndarray:
    """
    Build the 46-feature inference vector matching training order.
    """
    h  = ata_actual.hour + ata_actual.minute / 60.0
    dw = ata_actual.weekday()
    dm = ata_actual.day
    mo = ata_actual.month
    wk = int(ata_actual.isocalendar()[1])
    qt = (mo - 1) // 3 + 1

    is_wknd    = int(dw >= 5)
    is_peak    = int(8 <= ata_actual.hour <= 18)
    holiday    = int(ata_actual.date() in HOLIDAY_SET)
    d2hol      = _days_to_holiday(ata_actual.date())
    eta_dev    = (ata_actual - eta_planned).total_seconds() / 60.0
    eta_dev    = max(-120, min(480, eta_dev))

    tier       = COMPANY_TIER.get(company_name, 3)
    is_cont    = int(vessel_type == 'CONTAINER')
    teu_norm   = min(teu_capacity / 24_000, 1.0)
    dwt_norm   = min(dwt / 220_000, 1.0)
    loa_norm   = min(loa / 400, 1.0)

    teu_total  = teu_loaded + teu_discharged
    load_f     = (teu_total / max(teu_capacity, 1)) if teu_capacity > 0 else 0.0
    load_f     = min(load_f, 1.0)

    port_haifa = int(port_name == 'Haifa')
    berth_zone = berth_id[0] if berth_id else 'H'
    berth_num  = int(berth_id[1:]) if berth_id and berth_id[1:].isdigit() else 1

    bc_ratio   = arrivals_12h / max(BERTH_COUNTS.get(port_name, 15), 1)
    crane_risk = cranes_used * min(bc_ratio, 3.0)
    svc_freq   = 0.5  # default mid-frequency

    weather_storm = int(weather_wind_knots > 25)

    cargo_log  = math.log1p(cargo_tons)
    teu_imbal  = abs(teu_loaded - teu_discharged) / max(teu_total, 1)

    h_sin  = math.sin(2 * math.pi * h / 24)
    h_cos  = math.cos(2 * math.pi * h / 24)
    dw_sin = math.sin(2 * math.pi * dw / 7)
    dw_cos = math.cos(2 * math.pi * dw / 7)
    mo_sin = math.sin(2 * math.pi * mo / 12)
    mo_cos = math.cos(2 * math.pi * mo / 12)

    vtype_enc  = VESSEL_TYPE_ENC.get(vessel_type, 0)
    vclass_enc = _teu_class_enc(vessel_type, teu_capacity)
    port_enc   = PORT_ENC.get(port_name, 0)
    bzone_enc  = BERTH_ZONE_ENC.get(berth_zone, 0)
    svc_enc    = SERVICE_LINE_ENC.get(service_line, 0)

    vec = np.array([
        h, dw, dm, mo, wk, qt,
        is_wknd, is_peak, holiday, d2hol, eta_dev,
        tier, is_cont, teu_norm, dwt_norm, loa_norm,
        load_f, port_haifa, berth_num,
        arrivals_6h, arrivals_12h, arrivals_24h,
        bc_ratio, queue_position, crane_risk,
        svc_freq, weather_wind_knots, weather_storm,
        cargo_log, teu_imbal,
        h_sin, h_cos, dw_sin, dw_cos, mo_sin, mo_cos,
        cranes_used, dwt,
        weather_wind_knots, weather_storm, berth_competition,
        vtype_enc, vclass_enc, port_enc, bzone_enc, svc_enc,
    ], dtype=np.float32)

    return vec


# ─────────────────────────────────────────────────────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PortPredictor:
    """Wraps all three Phase 2 models for production inference."""

    def __init__(self):
        self._m1 = None
        self._m2 = None
        self._m3 = None

    def load(self):
        self._m1 = joblib.load(os.path.join(MODEL_DIR, "waiting_time_ensemble.pkl"))
        self._m2 = joblib.load(os.path.join(MODEL_DIR, "berth_occupancy.pkl"))
        self._m3 = joblib.load(os.path.join(MODEL_DIR, "congestion_risk.pkl"))
        return self

    # ── Model 1: Waiting time ─────────────────────────────────────────────────
    def predict_waiting_time(self, X: np.ndarray) -> tuple[float, float, float]:
        """Returns (point_estimate, ci_low, ci_high) in hours."""
        xgb_pred = float(self._m1['xgb_reg'].predict(X.reshape(1, -1))[0])
        lgb_pred = float(self._m1['lgb_reg'].predict(X.reshape(1, -1))[0])
        w = self._m1['ensemble_weight']
        point = max(0.0, w * xgb_pred + (1 - w) * lgb_pred)

        # Confidence interval: ±1.5 MAE from training metrics
        mae = self._m1['metrics']['mae']
        ci_lo = max(0.0, round(point - 1.5 * mae, 1))
        ci_hi = round(point + 1.5 * mae, 1)
        return round(point, 1), ci_lo, ci_hi

    # ── Model 2: Berth occupancy ──────────────────────────────────────────────
    def predict_occupancy(self, X: np.ndarray) -> dict:
        """Returns class label and probabilities."""
        proba = self._m2['model'].predict_proba(X.reshape(1, -1))[0]
        label_idx = int(proba.argmax())
        labels = self._m2['label_names']
        return {
            "occupancy_class": labels[label_idx],
            "probabilities": {l: round(float(p), 3) for l, p in zip(labels, proba)},
        }

    # ── Model 3: Congestion risk ──────────────────────────────────────────────
    def predict_congestion(self, X: np.ndarray) -> float:
        """Returns congestion probability score 0-1."""
        proba = float(self._m3['model'].predict_proba(X.reshape(1, -1))[0][1])
        return round(proba, 3)

    # ── Berth recommendation ─────────────────────────────────────────────────
    def recommend_berth(
        self, port_name: str, loa: int, draft: float, vessel_type: str
    ) -> str:
        """Simple rule-based berth recommender (placeholder for Phase 4 optimizer)."""
        zone = 'H' if port_name == 'Haifa' else 'A'
        max_berths = 20 if port_name == 'Haifa' else 15

        # Large vessels → higher berth numbers (deeper, longer quays)
        if loa > 300 or draft > 14.0:
            berth_num = np.random.randint(max_berths - 4, max_berths + 1)
        elif vessel_type in ('BULK', 'TANKER'):
            berth_num = np.random.randint(max_berths // 2, max_berths - 4)
        else:
            berth_num = np.random.randint(1, max_berths // 2 + 1)

        return f"{zone}{berth_num:02d}"

    # ── Hourly berth forecast (24h) ───────────────────────────────────────────
    def predict_hourly_berth(
        self, berth_id: str, forecast_date: date, port_name: str
    ) -> list[dict]:
        """
        Generate 24 hourly occupancy predictions for a given berth and date.
        Uses Model 2 with hour-varied feature vectors.
        """
        results = []
        base_dt = datetime.combine(forecast_date, datetime.min.time())

        # Typical background competition for that day
        dw = base_dt.weekday()
        day_mult = {0: 1.2, 1: 1.2, 2: 1.0, 3: 0.8, 4: 0.5, 5: 0.1, 6: 0.9}.get(dw, 1.0)
        bc = float(np.random.gamma(1.5 * day_mult, 0.7))

        for hour in range(24):
            ata = base_dt.replace(hour=hour)
            X = build_inference_features(
                eta_planned=ata,
                ata_actual=ata,
                port_name=port_name,
                vessel_type='CONTAINER',
                teu_capacity=8000,
                dwt=80000,
                loa=250,
                company_name='Maersk',
                service_line='Asia-EU',
                berth_id=berth_id,
                cranes_used=3,
                cargo_tons=80000,
                teu_loaded=4000,
                teu_discharged=4000,
                weather_wind_knots=float(np.random.exponential(8)),
                berth_competition=bc,
                arrivals_6h=int(5 * day_mult),
                arrivals_12h=int(10 * day_mult),
                arrivals_24h=int(20 * day_mult),
                queue_position=hour % 10 + 1,
            )
            occ = self.predict_occupancy(X)
            proba = occ['probabilities']
            util = proba.get('Low', 0) * 0.2 + proba.get('Medium', 0) * 0.6 + proba.get('High', 0) * 0.9
            results.append({
                "hour": hour,
                "utilization": round(util, 3),
                "occupancy_class": occ['occupancy_class'],
                "probabilities": proba,
            })
        return results


# Singleton — loaded once at startup
predictor = PortPredictor()
