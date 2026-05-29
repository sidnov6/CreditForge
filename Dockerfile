# ============================================================================
# CreditForge — single-container deploy (HF Spaces / any Docker host).
# FastAPI serves the API under /api and the pre-built static Next.js cockpit at
# /. Model artifacts are baked at build time by running the full pipeline once
# in the trainer stage, then copied into a slim runtime image (no train at boot).
# ============================================================================

# ---- Stage 1: build the static Next.js cockpit -----------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY app/dashboard/package.json app/dashboard/package-lock.json ./
RUN npm ci
COPY app/dashboard/ ./
# NEXT_PUBLIC_API_URL="" -> the cockpit calls the same-origin /api
RUN npm run build

# ---- Stage 2: train models + generate artifacts (full deps) -----------------
FROM python:3.11-slim AS trainer
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY creditforge/ ./creditforge/
ENV PYTHONPATH=/app
# the data/artifact dirs are .dockerignored (regenerable) — create them fresh
RUN mkdir -p creditforge/data/bronze creditforge/data/silver \
             creditforge/data/gold creditforge/artifacts creditforge/reports
# generate -> Silver -> Gold -> train -> validate -> governance -> monitor -> gates
# (a failing gate fails the build — exactly what we want)
RUN python -m creditforge.cli all

# ---- Stage 3: slim runtime image -------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# application code
COPY creditforge/ ./creditforge/
COPY app/ ./app/
# baked artifacts + the Gold matrix (serving needs it for the vintage rate lookup)
COPY --from=trainer /app/creditforge/artifacts/ ./creditforge/artifacts/
COPY --from=trainer /app/creditforge/data/gold/ ./creditforge/data/gold/
COPY --from=trainer /app/creditforge/governance/model_card.md ./creditforge/governance/model_card.md
# static cockpit
COPY --from=web /web/out/ ./static/

ENV PYTHONPATH=/app \
    STATIC_DIR=/app/static \
    PORT=7860
EXPOSE 7860

# HF Spaces routes traffic to port 7860 by default.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT}"]
