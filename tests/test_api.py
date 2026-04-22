"""
tests/test_api.py — Suite de pruebas para la API de detección de anomalías
"""
import json
import os
import sys

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app, state, FEATURES
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


client = TestClient(app)


@pytest.fixture(autouse=True)
def load_test_model(tmp_path):
    """Crea y carga un modelo de prueba antes de cada test."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, size=(300, len(FEATURES)))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  IsolationForest(n_estimators=50, contamination=0.05, random_state=42)),
    ])
    pipeline.fit(X)

    meta = {
        "version": "v_test",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "hyperparameters": {"contamination": 0.05, "n_estimators": 50},
        "metrics": {"total_records": 300, "anomalies": 15},
    }

    state["pipeline"] = pipeline
    state["metadata"] = meta

    yield

    state["pipeline"] = None
    state["metadata"] = None


NORMAL_PAYLOAD = {
    "cpu_usage": 45.0,
    "memory_usage": 50.0,
    "network_traffic": 500.0,
    "power_consumption": 250.0,
    "execution_time": 50.0,
    "energy_efficiency": 0.5,
}


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is True

    def test_health_degraded_when_no_model(self):
        state["pipeline"] = None
        r = client.get("/health")
        assert r.json()["status"] == "degraded"


class TestPredict:
    def test_predict_returns_required_fields(self):
        r = client.post("/predict", json=NORMAL_PAYLOAD)
        assert r.status_code == 200
        for field in ["anomaly", "label", "anomaly_score", "severity", "message", "timestamp"]:
            assert field in r.json()

    def test_label_is_1_or_minus1(self):
        r = client.post("/predict", json=NORMAL_PAYLOAD)
        assert r.json()["label"] in [1, -1]

    def test_anomaly_matches_label(self):
        r = client.post("/predict", json=NORMAL_PAYLOAD)
        body = r.json()
        assert body["anomaly"] == (body["label"] == -1)

    def test_503_when_no_model(self):
        state["pipeline"] = None
        r = client.post("/predict", json=NORMAL_PAYLOAD)
        assert r.status_code == 503

    def test_validation_cpu_out_of_range(self):
        bad = {**NORMAL_PAYLOAD, "cpu_usage": 150.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_validation_missing_required_field(self):
        bad = {k: v for k, v in NORMAL_PAYLOAD.items() if k != "cpu_usage"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestModelInfo:
    def test_returns_metadata(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        assert r.json()["version"] == "v_test"

    def test_404_when_no_metadata(self):
        state["metadata"] = None
        r = client.get("/model/info")
        assert r.status_code == 404


class TestPredictBatch:
    def test_batch_returns_correct_count(self):
        payload = {"records": [NORMAL_PAYLOAD, NORMAL_PAYLOAD, NORMAL_PAYLOAD]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert r.json()["total"] == 3

    def test_batch_empty_list_rejected(self):
        r = client.post("/predict/batch", json={"records": []})
        assert r.status_code == 422
