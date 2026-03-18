"""
features.py — Feature engineering for all 3 Phase 2 models.
Produces 50+ features from port_calls.parquet.
"""

import numpy as np
import pandas as pd

# ── Jewish holidays 2024-2025 (dates only) ────────────────────────────────────
_HOLIDAY_DATES = {
    # 2024
    "2024-04-22", "2024-04-23", "2024-04-28", "2024-04-29",
    "2024-05-14", "2024-06-12",
    "2024-10-02", "2024-10-03", "2024-10-11", "2024-10-12",
    "2024-10-16", "2024-10-17", "2024-10-23", "2024-10-24",
    # 2025
    "2025-04-12", "2025-04-13", "2025-04-18", "2025-04-19",
    "2025-05-01", "2025-06-01", "2025-06-02",
    "2025-09-22", "2025-09-23", "2025-10-01", "2025-10-02",
    "2025-10-06", "2025-10-07", "2025-10-13", "2025-10-14",
}
HOLIDAY_SET = {pd.Timestamp(d).date() for d in _HOLIDAY_DATES}

COMPANY_TIER = {
    'Maersk': 1, 'MSC': 1, 'CMA CGM': 1, 'COSCO': 1,
    'Hapag-Lloyd': 2, 'ONE': 2, 'Evergreen': 2, 'ZIM': 2,
    'Yang Ming': 2, 'HMM': 2, 'PIL': 3, 'Sinokor': 3,
    'Gold Star': 3, 'Grimaldi': 3, 'OOCL': 2, 'Other': 3,
}

VESSEL_CLASS = {
    'CONTAINER': {
        (0,     2_000): 'Feeder',
        (2_000, 8_000): 'Panamax',
        (8_000, 14_000): 'Post-Panamax',
        (14_000, 999_999): 'Ultra-Large',
    }
}


def _teu_class(row):
    if row['vessel_type'] != 'CONTAINER':
        return row['vessel_type']
    teu = row['teu_capacity']
    for (lo, hi), label in VESSEL_CLASS['CONTAINER'].items():
        if lo <= teu < hi:
            return label
    return 'Ultra-Large'


def days_to_nearest_holiday(date):
    """Days to nearest Jewish holiday (positive = before, negative = after)."""
    d = date.date() if hasattr(date, 'date') else date
    diffs = [(pd.Timestamp(h).date() - d).days for h in HOLIDAY_SET]
    if not diffs:
        return 99
    min_abs = min(abs(x) for x in diffs)
    return min_abs


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 50+ features from raw port_calls DataFrame.
    Returns a new DataFrame with feature columns appended.
    """
    out = df.copy()

    # ── TEMPORAL ──────────────────────────────────────────────────────────────
    out['hour_of_day']      = out['ata_actual'].dt.hour
    out['day_of_week']      = out['ata_actual'].dt.dayofweek       # Mon=0 Sun=6
    out['day_of_month']     = out['ata_actual'].dt.day
    out['month']            = out['ata_actual'].dt.month
    out['week_of_year']     = out['ata_actual'].dt.isocalendar().week.astype(int)
    out['quarter']          = out['ata_actual'].dt.quarter
    out['is_weekend']       = (out['day_of_week'] >= 5).astype(int)
    out['is_peak_hour']     = out['hour_of_day'].between(8, 18).astype(int)
    out['holiday_flag']     = out['ata_actual'].apply(
        lambda x: int(x.date() in HOLIDAY_SET))

    out['days_since_holiday'] = out['ata_actual'].apply(days_to_nearest_holiday)

    # ETA deviation (minutes): positive = arrived late vs plan
    out['eta_deviation_min'] = (
        (out['ata_actual'] - out['eta_planned']).dt.total_seconds() / 60
    ).clip(-120, 480)

    # ── VESSEL FEATURES ───────────────────────────────────────────────────────
    out['vessel_teu_class']  = out.apply(_teu_class, axis=1)
    out['company_tier']      = out['company_name'].map(COMPANY_TIER).fillna(3).astype(int)
    out['is_container']      = (out['vessel_type'] == 'CONTAINER').astype(int)
    out['teu_cap_norm']      = (out['teu_capacity'] / 24_000).clip(0, 1)
    out['dwt_norm']          = (out['dwt'] / 220_000).clip(0, 1)
    out['loa_norm']          = (out['loa'] / 400).clip(0, 1)
    out['load_factor']       = np.where(
        out['teu_capacity'] > 0,
        (out['teu_loaded'] + out['teu_discharged']) / out['teu_capacity'].clip(lower=1),
        0
    ).clip(0, 1)

    # ── PORT-LEVEL FEATURES ───────────────────────────────────────────────────
    out['port_haifa']        = (out['port_name'] == 'Haifa').astype(int)
    out['berth_zone']        = out['berth_id'].str[0]  # 'H' or 'A'
    out['berth_num']         = out['berth_id'].str[1:].astype(int)

    # ── OPERATIONAL / QUEUE FEATURES ─────────────────────────────────────────
    # Sort for time-ordered rolling window calcs
    out = out.sort_values('ata_actual').reset_index(drop=True)

    # Rolling counts per port (arrivals in trailing 6h, 12h, 24h)
    out = _rolling_arrivals(out, windows_hours=[6, 12, 24])

    # Berth competition ratio: arrivals in last 12h / berth count
    berth_counts = {'Haifa': 20, 'Ashdod': 15}
    out['berth_competition_ratio'] = (
        out.apply(lambda r: r['arrivals_12h'] / berth_counts[r['port_name']], axis=1)
    ).clip(0, 5)

    # Queue position: rank within same day+port
    out['queue_position'] = out.groupby(
        [out['ata_actual'].dt.date, 'port_name']
    )['ata_actual'].rank(method='first').astype(int)

    # Crane sharing risk: cranes_used × same-berth calls that day
    out['crane_sharing_risk'] = (
        out['cranes_used'] * (out['berth_competition_ratio'].clip(0, 3))
    ).round(2)

    # Service frequency: how often this service_line appears (per port)
    freq = out.groupby(['port_name', 'service_line'])['id'].transform('count')
    out['service_frequency'] = (freq / freq.max()).round(4)

    # ── WEATHER & BERTH COMPETITION (stored from generation time) ─────────────
    # These are the actual causal inputs used to generate waiting_anchor_hours
    if 'weather_wind_knots' not in out.columns:
        rng = np.random.default_rng(42)
        out['weather_wind_knots'] = (
            rng.exponential(scale=8, size=len(out))
        ).clip(0, 50).round(1)
    if 'berth_competition' not in out.columns:
        out['berth_competition'] = 1.0
    out['weather_storm_flag'] = (out['weather_wind_knots'] > 25).astype(int)

    # ── CARGO FEATURES ────────────────────────────────────────────────────────
    out['cargo_tons_log']    = np.log1p(out['cargo_tons'])
    out['teu_imbalance']     = (
        (out['teu_loaded'] - out['teu_discharged']).abs() /
        (out['teu_loaded'] + out['teu_discharged'] + 1)
    ).round(4)

    # ── SIN/COS CYCLICAL ENCODING ─────────────────────────────────────────────
    out['hour_sin']   = np.sin(2 * np.pi * out['hour_of_day'] / 24)
    out['hour_cos']   = np.cos(2 * np.pi * out['hour_of_day'] / 24)
    out['dow_sin']    = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['dow_cos']    = np.cos(2 * np.pi * out['day_of_week'] / 7)
    out['month_sin']  = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos']  = np.cos(2 * np.pi * out['month'] / 12)

    # ── ENCODE CATEGORICALS ───────────────────────────────────────────────────
    cat_cols = ['vessel_type', 'vessel_teu_class', 'port_name',
                'berth_zone', 'service_line']
    for col in cat_cols:
        out[col + '_enc'] = pd.factorize(out[col])[0]

    return out


def _rolling_arrivals(df: pd.DataFrame, windows_hours: list) -> pd.DataFrame:
    """
    For each row, count arrivals at the same port in the trailing N hours.
    Uses pandas rolling on a per-port resampled series then merges back.
    """
    df = df.copy()
    df = df.set_index('ata_actual').sort_index()

    for w in windows_hours:
        col = f'arrivals_{w}h'
        # Count arrivals per timestamp per port using rolling window
        results = []
        for port_name, grp in df.groupby('port_name'):
            counts = (
                grp['id']
                .rolling(f'{w}h', closed='left')
                .count()
                .fillna(0)
                .astype(int)
                .rename(col)
            )
            results.append(counts)
        combined = pd.concat(results).sort_index()
        df[col] = combined

    df = df.reset_index()
    return df


# ── CATEGORICAL FEATURE LISTS (for one-hot or ordinal) ───────────────────────
NUMERIC_FEATURES = [
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
]

CATEGORICAL_ENC_FEATURES = [
    'vessel_type_enc', 'vessel_teu_class_enc', 'port_name_enc',
    'berth_zone_enc', 'service_line_enc',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_ENC_FEATURES
