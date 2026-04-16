import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rossmann_ops.api.main import app


@pytest.fixture
def client():
    """Provides a TestClient with robustly mocked background artifacts."""
    mock_config = {
        "data": {"raw_store": "dummy"},
        "model": {"target_means_path": "dummy", "save_path": "dummy", "type": "RF"},
        "features": {
            "ohe_expected_columns": ["StoreType_a", "Assortment_a", "StateHoliday_0"]
        },
    }

    # 1. Mock all external dependencies during startup and execution
    with (
        patch("rossmann_ops.api.main.mlflow.sklearn.load_model") as mock_load_model,
        patch("builtins.open", MagicMock()),
        patch("rossmann_ops.api.main.yaml.safe_load", return_value=mock_config),
        patch("rossmann_ops.api.main.pd.read_csv") as mock_read_csv,
        patch("rossmann_ops.api.main.json.load") as mock_json_load,
    ):
        # Setup model mock
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([np.log1p(5000.0)])
        mock_load_model.return_value = mock_model

        # Setup store metadata mock
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Store": [1],
                "StoreType": ["a"],
                "Assortment": ["a"],
                "CompetitionDistance": [100.0],
            }
        )

        # Setup target means mock
        mock_json_load.return_value = {
            "store_means": {"1": 5000.0},
            "global_mean": 4500.0,
        }

        # 2. Inject state into the module to bypass failed startup or local missing files
        from rossmann_ops.api import main

        main.MODEL = mock_model
        main.STORE_DF = mock_read_csv.return_value
        main.STORE_MEANS = {1: 5000.0}
        main.GLOBAL_MEAN = 4500.0

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """GET /health must return 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_payload(client):
    """POST /predict with valid JSON must return 200 and a numeric PredictedSales."""
    payload = {
        "Store": 1,
        "DayOfWeek": 3,
        "Date": "2026-04-08",
        "Promo": 1,
        "StateHoliday": "0",
        "StoreType": "a",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "PredictedSales" in data
    assert isinstance(data["PredictedSales"], (float, int))


def test_predict_invalid_poison_payload(client):
    """POST /predict with absurd CompetitionDistance must return 422 (blocked)."""
    payload = {
        "Store": 1,
        "DayOfWeek": 3,
        "Date": "2026-04-08",
        "Promo": 1,
        "StateHoliday": "0",
        "StoreType": "a",
        "Assortment": "a",
        "CompetitionDistance": 999999.0,  # Poisoned/Anomalous
    }
    response = client.post("/predict", json=payload)
    # The API blocks > 100_000 with 422
    assert response.status_code == 422
    assert "blocked" in response.json()["detail"].lower()


def test_predict_missing_fields(client):
    """POST /predict with missing required fields must return 422 (Unprocessable Entity)."""
    payload = {"Store": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_latency(client):
    """POST /predict must respond within 100ms for mocked backend."""
    payload = {
        "Store": 1,
        "DayOfWeek": 3,
        "Date": "2026-04-08",
        "Promo": 1,
        "StateHoliday": "0",
        "StoreType": "a",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
    }

    start_time = time.perf_counter()
    response = client.post("/predict", json=payload)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    assert response.status_code == 200
    assert latency_ms < 100, f"API latency exceeded threshold: {latency_ms:.2f}ms"
