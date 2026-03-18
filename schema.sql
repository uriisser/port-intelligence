-- schema.sql: port_calls_synthetic DDL
-- Israeli Ports Synthetic Dataset, Jan 2024 - Dec 2025

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
    created_date          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_port_calls_port   ON port_calls_synthetic(port_name);
CREATE INDEX idx_port_calls_ata    ON port_calls_synthetic(ata_actual);
CREATE INDEX idx_port_calls_type   ON port_calls_synthetic(vessel_type);
CREATE INDEX idx_port_calls_imo    ON port_calls_synthetic(vessel_imo);
