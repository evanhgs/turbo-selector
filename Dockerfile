# ============================================
# Stage 1: Builder - Installation dépendances
# ============================================
FROM python:3.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# ============================================
# Stage 2: Runtime - Image finale légère
# ============================================
FROM python:3.13

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:\$PATH \
    WORKERS=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

WORKDIR /app

COPY --chown=appuser:appuser . ./app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port \${PORT} \
    --workers \${WORKERS} \
    --log-level info \
    --no-access-log \
    --proxy-headers \
    --forwarded-allow-ips=${ALLOWED_IP}