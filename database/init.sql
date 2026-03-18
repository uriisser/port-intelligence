-- database/init.sql
-- Port Intelligence DB — schema + seed for Docker entrypoint

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── 1. Raw port calls (from Phase 1 COPY dump) ───────────────────────────────
CREATE TABLE IF NOT EXISTS port_calls_synthetic (
    id                    SERIAL PRIMARY KEY,
    port_name             VARCHAR(10)  NOT NULL CHECK (port_name IN ('Haifa','Ashdod')),
    vessel_imo            VARCHAR(8),
    vessel_name           VARCHAR(50),
    vessel_type           VARCHAR(20),
    dwt                   INTEGER      CHECK (dwt > 0),
    teu_capacity          INTEGER,
    loa                   INTEGER      CHECK (loa BETWEEN 50 AND 450),
    draft                 DECIMAL(4,1) CHECK (draft BETWEEN 2.0 AND 18.0),
    company_name          VARCHAR(50),
    service_line          VARCHAR(30),
    eta_planned           TIMESTAMP NOT NULL,
    ata_actual            TIMESTAMP NOT NULL,
    atb                   TIMESTAMP,
    etd                   TIMESTAMP,
    atd_actual            TIMESTAMP NOT NULL,
    waiting_anchor_hours  DECIMAL(5,1) GENERATED ALWAYS AS
                              (EXTRACT(EPOCH FROM (atb - ata_actual))/3600) STORED,
    waiting_berth_hours   DECIMAL(5,1) GENERATED ALWAYS AS
                              (EXTRACT(EPOCH FROM (etd - atb))/3600) STORED,
    berth_id              VARCHAR(20),
    cranes_used           INTEGER      CHECK (cranes_used BETWEEN 0 AND 8),
    cargo_tons            DECIMAL(10,0),
    teu_loaded            INTEGER      DEFAULT 0,
    teu_discharged        INTEGER      DEFAULT 0,
    weather_wind_knots    DECIMAL(4,1),
    berth_competition     DECIMAL(5,3),
    created_date          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pc_port   ON port_calls_synthetic(port_name);
CREATE INDEX IF NOT EXISTS idx_pc_ata    ON port_calls_synthetic(ata_actual);
CREATE INDEX IF NOT EXISTS idx_pc_type   ON port_calls_synthetic(vessel_type);
CREATE INDEX IF NOT EXISTS idx_pc_imo    ON port_calls_synthetic(vessel_imo);
CREATE INDEX IF NOT EXISTS idx_pc_berth  ON port_calls_synthetic(berth_id);

-- ── 2. Live API prediction log ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vessel_predictions (
    prediction_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vessel_id             VARCHAR(50) NOT NULL,
    port_name             VARCHAR(10),
    predicted_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    eta_planned           TIMESTAMP,
    waiting_anchor_forecast DECIMAL(5,1),
    ci_low                DECIMAL(5,1),
    ci_high               DECIMAL(5,1),
    recommended_berth     VARCHAR(10),
    congestion_risk       DECIMAL(5,3),
    congestion_flag       BOOLEAN,
    occupancy_class       VARCHAR(10),
    model_version         VARCHAR(30) DEFAULT 'phase2-v1',
    -- Actuals filled in after arrival
    actual_waiting_hours  DECIMAL(5,1),
    mae_error             DECIMAL(5,1)
);

CREATE INDEX IF NOT EXISTS idx_vp_vessel ON vessel_predictions(vessel_id);
CREATE INDEX IF NOT EXISTS idx_vp_port   ON vessel_predictions(port_name);
CREATE INDEX IF NOT EXISTS idx_vp_at     ON vessel_predictions(predicted_at);

-- ── 3. Berth hourly forecast store ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hourly_berth_forecast (
    forecast_id    SERIAL PRIMARY KEY,
    berth_id       VARCHAR(10) NOT NULL,
    port_name      VARCHAR(10) NOT NULL,
    forecast_date  DATE        NOT NULL,
    hour           SMALLINT    NOT NULL CHECK (hour BETWEEN 0 AND 23),
    utilization    DECIMAL(4,3),
    occupancy_class VARCHAR(10),
    prob_low       DECIMAL(4,3),
    prob_medium    DECIMAL(4,3),
    prob_high      DECIMAL(4,3),
    created_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (berth_id, forecast_date, hour)
);

CREATE INDEX IF NOT EXISTS idx_hbf_berth ON hourly_berth_forecast(berth_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_hbf_date  ON hourly_berth_forecast(forecast_date);
