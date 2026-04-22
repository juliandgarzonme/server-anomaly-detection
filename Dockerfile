# ── Stage 1: entrenamiento ──────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y datos
COPY scripts/ scripts/
COPY data/    data/

# Entrenar modelo — genera models/isolation_forest.pkl y models/metadata.json
RUN mkdir -p models && \
    python scripts/train.py \
        --data data/intel_dataset.csv \
        --contamination 0.05 \
        --n_estimators 200 \
        --random_state 42

# ── Stage 2: API (imagen final liviana) ────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo lo necesario desde el stage de entrenamiento
COPY --from=trainer /app/models/ models/

# Código de la API
COPY main.py .

# Usuario no-root para seguridad
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Puerto de escucha
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Arrancar servidor
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
