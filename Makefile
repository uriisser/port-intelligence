# ── Port Intelligence — Makefile ──────────────────────────────────────────────
# Usage:
#   make up        Build images and start all services
#   make down      Stop and remove containers
#   make test      Run API smoke tests
#   make logs      Tail service logs
#   make shell     Open API container shell
#   make db        Open psql prompt

COMPOSE   = docker-compose
API_URL   = http://localhost:8000
DASH_URL  = http://localhost:8501

.PHONY: up down restart test logs shell db clean build \
        run-api run-dashboard phase1 phase2

# ── Docker operations ─────────────────────────────────────────────────────────

up: build
	$(COMPOSE) up -d
	@echo ""
	@echo "Services starting..."
	@echo "  API:       $(API_URL)/docs"
	@echo "  Dashboard: $(DASH_URL)"
	@echo "  Postgres:  localhost:5432"
	@echo ""
	@echo "Run 'make test' to verify the stack."

build:
	$(COMPOSE) build --no-cache api

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart api

logs:
	$(COMPOSE) logs -f api

logs-all:
	$(COMPOSE) logs -f

shell:
	$(COMPOSE) exec api bash

db:
	$(COMPOSE) exec postgres psql -U portuser -d port_intelligence


# ── Smoke tests (no external deps, uses curl) ─────────────────────────────────

test: test-health test-predict test-berth test-ports
	@echo ""
	@echo "All smoke tests passed."

test-health:
	@echo "--- /health ---"
	@curl -sf $(API_URL)/health | python -m json.tool
	@echo ""

test-predict:
	@echo "--- POST /predict_vessel ---"
	@curl -sf -X POST $(API_URL)/predict_vessel \
	  -H "Content-Type: application/json" \
	  -d '{ \
	    "vessel_id": "TEST-IMO-001", \
	    "port_name": "Haifa", \
	    "eta_planned": "2025-12-01T10:00:00", \
	    "vessel_type": "CONTAINER", \
	    "teu_capacity": 8000, \
	    "dwt": 80000, \
	    "loa": 250, \
	    "draft": 12.0, \
	    "company_name": "Maersk", \
	    "service_line": "Asia-EU", \
	    "berth_id": "H01", \
	    "cranes_used": 3, \
	    "cargo_tons": 80000, \
	    "teu_loaded": 4000, \
	    "teu_discharged": 4000, \
	    "weather_wind_knots": 8.0, \
	    "berth_competition": 1.2, \
	    "arrivals_6h": 5, \
	    "arrivals_12h": 10, \
	    "arrivals_24h": 20, \
	    "queue_position": 5 \
	  }' | python -m json.tool
	@echo ""

test-berth:
	@echo "--- GET /berth_forecast ---"
	@curl -sf "$(API_URL)/berth_forecast/H01/2025-12-01?port_name=Haifa" \
	  | python -c "import json,sys; d=json.load(sys.stdin); \
	    print('Berth:', d['berth_id'], '| Date:', d['forecast_date']); \
	    [print(f'  {p[\"hour\"]:02d}:00  {p[\"occupancy_class\"]:8s}  util={p[\"utilization\"]:.2f}') \
	     for p in d['predictions'][:6]]"
	@echo "  ... (24 hours total)"
	@echo ""

test-ports:
	@echo "--- GET /ports ---"
	@curl -sf $(API_URL)/ports | python -m json.tool
	@echo ""

test-metrics:
	@echo "--- GET /metrics ---"
	@curl -sf $(API_URL)/metrics | python -m json.tool
	@echo ""


# ── Local development (no Docker) ────────────────────────────────────────────

run-api:
	@echo "Starting API locally on $(API_URL)..."
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	@echo "Starting dashboard on $(DASH_URL)..."
	streamlit run demo/streamlit_app.py --server.port 8501

# Run both in background (requires two terminals or use tmux)
run-local:
	@echo "Start in two terminals:"
	@echo "  Terminal 1: make run-api"
	@echo "  Terminal 2: make run-dashboard"


# ── Data pipeline ─────────────────────────────────────────────────────────────

phase1:
	@echo "Running Phase 1: generating synthetic dataset..."
	python generate_data.py

phase2:
	@echo "Running Phase 2: training ML models..."
	python train_models.py


# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
