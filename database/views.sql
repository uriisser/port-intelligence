-- database/views.sql
-- Analytical views for the Port Intelligence system

-- ── 1. forecast_port_calls — daily vessel prediction summary ──────────────────
CREATE OR REPLACE VIEW forecast_port_calls AS
SELECT
    DATE(predicted_at)                        AS forecast_day,
    port_name,
    COUNT(*)                                  AS total_predictions,
    AVG(waiting_anchor_forecast)              AS avg_forecast_wait_h,
    PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY waiting_anchor_forecast)    AS median_forecast_wait_h,
    PERCENTILE_CONT(0.8) WITHIN GROUP
        (ORDER BY waiting_anchor_forecast)    AS p80_forecast_wait_h,
    COUNT(*) FILTER (WHERE congestion_flag)   AS congestion_alerts,
    ROUND(100.0 * COUNT(*) FILTER (WHERE congestion_flag) / NULLIF(COUNT(*),0), 1)
                                              AS congestion_pct,
    -- Accuracy (when actuals available)
    COUNT(*) FILTER (WHERE actual_waiting_hours IS NOT NULL)
                                              AS actuals_received,
    ROUND(AVG(ABS(actual_waiting_hours - waiting_anchor_forecast))::NUMERIC, 2)
                                              AS mae_hours,
    model_version
FROM vessel_predictions
GROUP BY 1, 2, model_version
ORDER BY 1 DESC, 2;

-- ── 2. hourly_berth_forecast view — latest forecast per berth/hour ────────────
CREATE OR REPLACE VIEW latest_berth_forecast AS
SELECT DISTINCT ON (berth_id, forecast_date, hour)
    berth_id,
    port_name,
    forecast_date,
    hour,
    utilization,
    occupancy_class,
    prob_low,
    prob_medium,
    prob_high,
    created_at
FROM hourly_berth_forecast
ORDER BY berth_id, forecast_date, hour, created_at DESC;

-- ── 3. Port throughput summary (historical data) ──────────────────────────────
CREATE OR REPLACE VIEW port_monthly_throughput AS
SELECT
    DATE_TRUNC('month', ata_actual)           AS month,
    port_name,
    COUNT(*)                                  AS total_calls,
    SUM(teu_loaded + teu_discharged)          AS total_teu,
    SUM(cargo_tons)                           AS total_cargo_tons,
    AVG(waiting_anchor_hours)                 AS avg_wait_h,
    PERCENTILE_CONT(0.8) WITHIN GROUP
        (ORDER BY waiting_anchor_hours)       AS p80_wait_h,
    PERCENTILE_CONT(0.95) WITHIN GROUP
        (ORDER BY waiting_anchor_hours)       AS p95_wait_h,
    COUNT(*) FILTER (
        WHERE vessel_type = 'CONTAINER')      AS container_calls,
    COUNT(*) FILTER (
        WHERE vessel_type = 'BULK')           AS bulk_calls,
    AVG(berth_competition)                    AS avg_berth_competition
FROM port_calls_synthetic
GROUP BY 1, 2
ORDER BY 1 DESC, 2;

-- ── 4. Congestion pattern analysis ───────────────────────────────────────────
CREATE OR REPLACE VIEW congestion_patterns AS
SELECT
    port_name,
    EXTRACT(DOW FROM ata_actual)::INT         AS day_of_week,
    EXTRACT(HOUR FROM ata_actual)::INT        AS hour_of_day,
    COUNT(*)                                  AS call_count,
    AVG(waiting_anchor_hours)                 AS avg_wait_h,
    AVG(berth_competition)                    AS avg_competition,
    COUNT(*) FILTER (
        WHERE waiting_anchor_hours >= 11.1)   AS congestion_count,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE waiting_anchor_hours >= 11.1
    ) / NULLIF(COUNT(*), 0), 1)              AS congestion_pct
FROM port_calls_synthetic
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- ── 5. Model accuracy tracking ────────────────────────────────────────────────
CREATE OR REPLACE VIEW model_accuracy_daily AS
SELECT
    DATE(predicted_at)                        AS eval_date,
    port_name,
    model_version,
    COUNT(*) FILTER (WHERE actual_waiting_hours IS NOT NULL) AS n_evaluated,
    ROUND(AVG(mae_error)::NUMERIC, 2)         AS avg_mae_h,
    ROUND(STDDEV(mae_error)::NUMERIC, 2)      AS mae_std_h,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE ABS(mae_error) < 2.0
    ) / NULLIF(COUNT(*) FILTER (WHERE actual_waiting_hours IS NOT NULL), 0), 1)
                                              AS pct_within_2h
FROM vessel_predictions
WHERE actual_waiting_hours IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1 DESC;
