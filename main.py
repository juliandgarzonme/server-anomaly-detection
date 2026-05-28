"""
main.py — API FastAPI para detección de anomalías en servidores
===============================================================
Endpoints:
    POST /predict          → clasifica una observación
    POST /predict/batch    → clasifica múltiples observaciones
    GET  /health           → estado del servicio y del modelo
    GET  /model/info       → metadatos del modelo cargado
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Rutas ────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_forest.pkl")
META_PATH  = os.getenv("META_PATH",  "models/metadata.json")

# ── Estado global del modelo ─────────────────────────────────────────
state: dict = {"pipeline": None, "metadata": None}


def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Modelo no encontrado en {MODEL_PATH}.")
        return
    artefacto = joblib.load(MODEL_PATH)
    # El artefacto puede ser un dict {model, scaler, features} o un pipeline directo
    if isinstance(artefacto, dict):
        from sklearn.pipeline import Pipeline
        modelo  = artefacto["model"]
        scaler  = artefacto["scaler"]
        # Reconstruir pipeline compatible con la API
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model",  modelo),
        ])
        state["pipeline"] = pipeline
        log.info(f"Artefacto cargado como dict — modelo: {type(modelo).__name__}")
    else:
        state["pipeline"] = artefacto
    log.info(f"Modelo listo: {MODEL_PATH}")

    if os.path.exists(META_PATH):
        import json
        with open(META_PATH, encoding="utf-8") as f:
            state["metadata"] = json.load(f)
        log.info(f"Metadatos cargados: versión {state['metadata'].get('version')}")
    else:
        log.warning(f"Metadatos no encontrados en {META_PATH}")


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    log.info("API detenida")


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Server Anomaly Detection API",
    description="Detección de anomalías en métricas de infraestructura usando Isolation Forest",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────────────
FEATURES = [
    "cpu_usage",
    "memory_usage",
    "network_traffic",
    "execution_time",
    "energy_efficiency",
]


class MetricsInput(BaseModel):
    cpu_usage:          float = Field(..., ge=0, le=100,    description="Uso de CPU (%)")
    memory_usage:       float = Field(..., ge=0, le=100,    description="Uso de memoria RAM (%)")
    network_traffic:    float = Field(..., ge=0,            description="Tráfico de red")
    power_consumption:  float = Field(..., ge=0,            description="Consumo energético")
    execution_time:     float = Field(default=50.0, ge=0,   description="Tiempo de ejecución (opcional)")
    energy_efficiency:  float = Field(default=0.5,  ge=0, le=1, description="Eficiencia energética (opcional)")

    @field_validator("cpu_usage", "memory_usage")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("El valor debe estar entre 0 y 100")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_usage": 95.5,
                "memory_usage": 88.2,
                "network_traffic": 950.0,
                "power_consumption": 480.0,
                "execution_time": 45.0,
                "energy_efficiency": 0.12,
            }
        }
    }


class PredictionResponse(BaseModel):
    anomaly:       bool
    label:         int          # 1 = normal, -1 = anomalía
    anomaly_score: float        # score continuo del modelo
    severity:      str          # "normal" | "warning" | "critical"
    message:       str
    timestamp:     str
    input_metrics: dict


class BatchInput(BaseModel):
    records: list[MetricsInput] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    total:          int
    anomalies:      int
    anomaly_rate:   float
    predictions:    list[PredictionResponse]


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: Optional[str]
    timestamp:     str


# ── Helpers ──────────────────────────────────────────────────────────
def severity_from_score(score: float, label: int) -> str:
    """Clasifica la severidad según el anomaly score."""
    if label == 1:
        return "normal"
    if score < -0.58:
        return "critical"
    return "warning"


def predict_single(metrics: MetricsInput) -> PredictionResponse:
    """Ejecuta la inferencia para una observación."""
    pipeline = state["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta el entrenamiento primero.")

    X = np.array([[
        metrics.cpu_usage,
        metrics.memory_usage,
        metrics.network_traffic,
        metrics.execution_time,
        metrics.energy_efficiency,
    ]])

    label = int(pipeline.predict(X)[0])
    score = float(pipeline.decision_function(X)[0])
    is_anomaly = label == -1
    sev = severity_from_score(score, label)

    if sev == "critical":
        msg = f"⚠️ ANOMALÍA CRÍTICA detectada (score={score:.4f}). Revisar infraestructura inmediatamente."
    elif sev == "warning":
        msg = f"⚠️ Comportamiento anómalo detectado (score={score:.4f}). Monitorear de cerca."
    else:
        msg = f"✅ Comportamiento normal (score={score:.4f})."

    log.info(f"Predicción: label={label} score={score:.4f} severity={sev} "
             f"cpu={metrics.cpu_usage} mem={metrics.memory_usage}")

    return PredictionResponse(
        anomaly=is_anomaly,
        label=label,
        anomaly_score=round(score, 6),
        severity=sev,
        message=msg,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_metrics=metrics.model_dump(),
    )


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Estado del servicio y disponibilidad del modelo."""
    version = None
    if state["metadata"]:
        version = state["metadata"].get("version")
    return HealthResponse(
        status="ok" if state["pipeline"] is not None else "degraded",
        model_loaded=state["pipeline"] is not None,
        model_version=version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Metadatos completos del modelo actualmente cargado."""
    if state["metadata"] is None:
        raise HTTPException(status_code=404, detail="Metadatos del modelo no disponibles.")
    return state["metadata"]


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(metrics: MetricsInput):
    """
    Clasifica una observación de métricas del servidor.

    - **anomaly**: true si el comportamiento es anómalo
    - **label**: 1 = normal, -1 = anomalía
    - **anomaly_score**: score continuo (más negativo = más anómalo)
    - **severity**: normal | warning | critical
    """
    return predict_single(metrics)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(body: BatchInput):
    """
    Clasifica múltiples observaciones en una sola llamada (máx. 500).
    """
    predictions = [predict_single(r) for r in body.records]
    anomalies = sum(1 for p in predictions if p.anomaly)
    return BatchResponse(
        total=len(predictions),
        anomalies=anomalies,
        anomaly_rate=round(anomalies / len(predictions) * 100, 2),
        predictions=predictions,
    )


@app.post("/model/reload", tags=["Modelo"])
def reload_model():
    """Recarga el modelo desde disco sin reiniciar el servicio."""
    load_model()
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="No se pudo cargar el modelo.")
    return {"status": "ok", "message": "Modelo recargado exitosamente."}
