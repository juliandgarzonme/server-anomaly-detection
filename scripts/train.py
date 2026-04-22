"""
train.py — Entrenamiento del modelo Isolation Forest
=====================================================
Uso:
    python scripts/train.py --data data/intel_dataset.csv --contamination 0.05

Genera:
    models/isolation_forest.pkl   → modelo + scaler empaquetados
    models/metadata.json          → metadatos del experimento
"""

import argparse
import json
import hashlib
import os
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────
FEATURES = [
    "cpu_usage",
    "memory_usage",
    "network_traffic",
    "power_consumption",
    "execution_time",
    "energy_efficiency",
]

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")


# ── Helpers ──────────────────────────────────────────────────────────
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_clean(path: str) -> pd.DataFrame:
    log.info(f"Cargando dataset: {path}")
    df = pd.read_csv(path)
    log.info(f"  → {df.shape[0]} registros, {df.shape[1]} columnas")

    # Conservar solo filas con todas las features presentes
    before = len(df)
    df = df.dropna(subset=FEATURES)
    after = len(df)
    log.info(f"  → {before - after} filas eliminadas por nulos | {after} filas limpias")

    return df[FEATURES]


def build_pipeline(contamination: float, n_estimators: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def evaluate(pipeline: Pipeline, X: pd.DataFrame) -> dict:
    labels = pipeline.predict(X)           # 1 = normal, -1 = anomalía
    scores = pipeline.decision_function(X) # score continuo

    n_anomalies = int((labels == -1).sum())
    n_normal    = int((labels ==  1).sum())

    return {
        "total_records":   len(X),
        "normal":          n_normal,
        "anomalies":       n_anomalies,
        "anomaly_rate_pct": round(n_anomalies / len(X) * 100, 2),
        "score_mean":      round(float(scores.mean()), 6),
        "score_std":       round(float(scores.std()),  6),
        "score_min":       round(float(scores.min()),  6),
        "score_max":       round(float(scores.max()),  6),
        "score_p5":        round(float(np.percentile(scores, 5)),  6),
        "score_p95":       round(float(np.percentile(scores, 95)), 6),
    }


def save_metadata(args, data_path: str, metrics: dict, version: str) -> None:
    import sklearn, sys
    meta = {
        "version":           version,
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path":          data_path,
            "md5":           file_md5(data_path),
            "features":      FEATURES,
        },
        "hyperparameters": {
            "contamination": args.contamination,
            "n_estimators":  args.n_estimators,
            "random_state":  args.random_state,
        },
        "metrics":           metrics,
        "environment": {
            "python":        sys.version,
            "sklearn":       sklearn.__version__,
        },
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadatos guardados: {META_PATH}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Isolation Forest")
    parser.add_argument("--data",          default="data/intel_dataset.csv", help="Ruta al CSV de entrenamiento")
    parser.add_argument("--contamination", type=float, default=0.05,  help="Proporción esperada de anomalías (0.01–0.5)")
    parser.add_argument("--n_estimators",  type=int,   default=200,   help="Número de árboles")
    parser.add_argument("--random_state",  type=int,   default=42,    help="Semilla aleatoria")
    args = parser.parse_args()

    # Versión basada en timestamp
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    log.info(f"Iniciando entrenamiento — versión: {version}")

    # Cargar y limpiar
    X = load_and_clean(args.data)

    # Construir y entrenar pipeline
    pipeline = build_pipeline(args.contamination, args.n_estimators, args.random_state)
    log.info(f"Entrenando Isolation Forest (n_estimators={args.n_estimators}, contamination={args.contamination})...")
    pipeline.fit(X)
    log.info("  → Entrenamiento completado")

    # Evaluar
    metrics = evaluate(pipeline, X)
    log.info(f"  → Anomalías detectadas: {metrics['anomalies']} ({metrics['anomaly_rate_pct']}%)")
    log.info(f"  → Score promedio: {metrics['score_mean']} | rango: [{metrics['score_min']}, {metrics['score_max']}]")

    # Guardar modelo y metadatos
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    log.info(f"Modelo guardado: {MODEL_PATH}")
    save_metadata(args, args.data, metrics, version)

    log.info("✅ Entrenamiento finalizado exitosamente")
    return metrics


if __name__ == "__main__":
    main()
