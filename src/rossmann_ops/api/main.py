import logging
import os
from pathlib import Path

import mlflow.xgboost
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from rossmann_ops.api.schemas import DriftRequest, PredictRequest, PredictResponse
from rossmann_ops.features import build_features

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Production API for predicting store sales using XGBoost.",
    version="1.0.0",
)

# Prometheus auto-instrumentation: exposes /metrics with RPS, latency, error rate.
Instrumentator().instrument(app).expose(app)

# Global model and config holders
MODEL = None
STORE_DF = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0-baseline")
PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)  # src/rossmann_ops/api/ -> rossmann_ops/ -> src/ -> repo root

# Custom business metrics
SALES_INFERENCE_TOTAL = Counter(
    "sales_inference_total",
    "Total successful sales predictions served.",
)
INFERENCE_ANOMALIES_BLOCKED = Counter(
    "inference_anomalies_blocked",
    "Total prediction requests blocked due to anomalous/poisoned input.",
)


@app.on_event("startup")
def load_artifacts():
    global MODEL, STORE_DF

    # 1. Load Parameters and Config
    try:
        config_path = PROJECT_ROOT / "configs" / "params.yaml"
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)

        # 2. Load Store data for feature enrichment
        store_path = PROJECT_ROOT / params["data"]["raw_store"]
        if store_path.exists():
            STORE_DF = pd.read_csv(store_path)
            logger.info("Successfully loaded store metadata.")
        else:
            logger.error(f"Store metadata not found at {store_path}")

        # 3. Load Model with Remote Registry capability & Graceful Local Fallback
        model_uri = os.getenv("MLFLOW_MODEL_URI")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if model_uri and tracking_uri:
            try:
                logger.info(f"Attempting dynamic model load from registry: {model_uri}")
                mlflow.set_tracking_uri(tracking_uri)
                MODEL = mlflow.xgboost.load_model(model_uri)
                logger.info("Successfully loaded model from remote registry.")
            except Exception as e:
                logger.warning(
                    f"Failed to load from registry, falling back to local: {e}"
                )

        if MODEL is None:
            model_path = PROJECT_ROOT / "models" / "production_model"
            if model_path.exists():
                MODEL = mlflow.xgboost.load_model(str(model_path))
                logger.info(f"Production model loaded from {model_path}")
            else:
                logger.warning(
                    "No model found dynamically or locally. Predict endpoint disabled."
                )

    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/health/live")
def liveness_check():
    """
    Returns 200 if the server process is running.
    Used for Kubernetes Liveness Probes.
    """
    return {"status": "alive"}


@app.get("/health")
def readiness_check(response: Response):
    """
    Returns system health and model status.
    Returns HTTP 503 if the service is not ready.
    Used for Kubernetes Readiness Probes.
    """
    is_ready = STORE_DF is not None and MODEL is not None
    if not is_ready:
        response.status_code = 503

    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": MODEL is not None,
        "store_data_loaded": STORE_DF is not None,
        "model_version": MODEL_VERSION,
        "environment": os.getenv("ENV", "development"),
    }


@app.get("/health/shap")
def get_shap_plot():
    """
    Serves the SHAP summary plot for model explainability.
    """
    shap_path = PROJECT_ROOT / "models" / "shap_summary.png"
    if not shap_path.exists():
        raise HTTPException(
            status_code=404, detail="Explainability artifacts not generated yet."
        )
    return FileResponse(shap_path)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Transaction-level sales prediction.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run 'just export-model' or ensure models/production_model exists.",
        )

    # Data-poisoning defence: hard boundary on CompetitionDistance.
    # Values above 50 km are physically implausible and indicate a malicious payload.
    if request.CompetitionDistance is not None and request.CompetitionDistance > 50_000:
        INFERENCE_ANOMALIES_BLOCKED.inc()
        logger.warning(
            f"Anomalous CompetitionDistance={request.CompetitionDistance} blocked for Store={request.Store}."
        )
        raise HTTPException(
            status_code=422,
            detail="CompetitionDistance exceeds plausible range (>50 000m). Request blocked.",
        )

    try:
        # 1. Convert request to DataFrame
        input_data = pd.DataFrame([request.model_dump()])

        # 2. Enrich with Store data
        # We merge on Store ID to get Competition/Store details
        enriched_df = pd.merge(
            input_data, STORE_DF, on="Store", how="left", suffixes=("", "_store")
        )

        # Override with request value if provided
        if request.CompetitionDistance is not None:
            enriched_df["CompetitionDistance"] = request.CompetitionDistance

        # 3. Applied Feature Engineering
        # We use the same build_features logic used during training
        processed_df = build_features(enriched_df)

        # 4. Feature selection to match training set
        # Note: We must ensure columns match exactly what xgb regressor expects
        # These are matched to src/rossmann_ops/train_model.py
        feature_cols = [
            "Store",
            "DayOfWeek",
            "Promo",
            "Year",
            "Month",
            "Day",
            "WeekOfYear",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
        ]

        # Check for missing columns (e.g. if store_df was missing data)
        for col in feature_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0  # Default fallback

        # 5. Inference
        prediction = MODEL.predict(processed_df[feature_cols])

        SALES_INFERENCE_TOTAL.inc()
        return {
            "Store": request.Store,
            "Date": request.Date,
            "PredictedSales": float(prediction[0]),
            "ModelVersion": MODEL_VERSION,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Inference error: {str(e)}"
        ) from None


@app.post("/drift-trigger")
def drift_trigger(request: DriftRequest):
    """
    Stub for Phase 6 Retraining trigger.
    """
    # Simply log the event for now
    logger.info(f"Drift check triggered for {len(request.sales_data)} records.")
    return {
        "drift_detected": False,
        "timestamp": pd.Timestamp.now().isoformat(),
        "message": "Drift monitoring stub active.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
