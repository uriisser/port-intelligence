# ── Stage 1: base Python ──────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for numpy/scipy/lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ── Stage 2: deps ─────────────────────────────────────────────────────────────
FROM base AS deps

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: final ────────────────────────────────────────────────────────────
FROM deps AS final

# Copy project source
COPY features.py          /app/features.py
COPY api/                 /app/api/
COPY models/              /app/models/

# Non-root user for security
RUN useradd -m -u 1001 portapi && chown -R portapi:portapi /app
USER portapi

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
