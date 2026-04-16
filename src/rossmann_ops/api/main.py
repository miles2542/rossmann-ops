import json
import logging
import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from rossmann_ops.api.schemas import DriftRequest, PredictRequest, PredictResponse
from rossmann_ops.features import apply_target_encoding, build_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Production API for predicting store sales using a Random Forest model.",
    version="1.0.0",
)

Instrumentator().instrument(
    app,
    latency_highr_buckets=(
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )
).expose(app)

# Global artifacts
MODEL = None
STORE_DF = None
STORE_MEANS: dict | None = None
GLOBAL_MEAN: float | None = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    global MODEL, STORE_DF, STORE_MEANS, GLOBAL_MEAN

    try:
        config_path = PROJECT_ROOT / "configs" / "params.yaml"
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)

        # 1. Load store metadata for CompetitionDistance enrichment
        store_path = PROJECT_ROOT / params["data"]["raw_store"]
        if store_path.exists():
            STORE_DF = pd.read_csv(store_path)
            logger.info("Store metadata loaded from %s.", store_path)
        else:
            logger.error("Store metadata not found at %s.", store_path)

        # 2. Load store target means artifact
        means_path = PROJECT_ROOT / params["model"]["target_means_path"]
        if means_path.exists():
            with open(means_path, "r") as f:
                artifact = json.load(f)
            # JSON keys are strings — convert back to int for Store ID lookup
            STORE_MEANS = {int(k): v for k, v in artifact["store_means"].items()}
            GLOBAL_MEAN = artifact["global_mean"]
            logger.info(
                "Store target means loaded. Stores covered: %d.", len(STORE_MEANS)
            )
        else:
            logger.error(
                "store_target_means.json not found at %s. Run training first.",
                means_path,
            )

        # 3. Load model from registry with local fallback
        model_uri = os.getenv("MLFLOW_MODEL_URI")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if model_uri and tracking_uri:
            try:
                logger.info("Loading model from registry: %s", model_uri)
                mlflow.set_tracking_uri(tracking_uri)
                MODEL = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded from remote registry.")
            except Exception as e:
                logger.warning("Registry load failed, falling back to local: %s", e)

        if MODEL is None:
            model_path = PROJECT_ROOT / params["model"]["save_path"]
            if model_path.exists():
                MODEL = mlflow.sklearn.load_model(str(model_path))
                logger.info("Model loaded from local path: %s.", model_path)
            else:
                logger.warning("No model found. Run 'just train-prod' first.")

    except Exception as e:
        logger.error("Startup failed: %s", e)


@app.get("/health/live")
def liveness_check():
    """Returns 200 if the server process is running. Used for K8s Liveness Probe."""
    return {"status": "alive"}


@app.get("/health")
def readiness_check(response: Response):
    """
    Returns system health and artifact status.
    Returns HTTP 503 if model, store data, or target means are not loaded.
    Used for K8s Readiness Probe.
    """
    is_ready = MODEL is not None and STORE_DF is not None and STORE_MEANS is not None
    if not is_ready:
        response.status_code = 503

    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": MODEL is not None,
        "store_data_loaded": STORE_DF is not None,
        "store_means_loaded": STORE_MEANS is not None,
        "model_version": MODEL_VERSION,
        "environment": os.getenv("ENV", "development"),
    }


@app.get("/health/shap")
def get_shap_plot():
    """Serves the SHAP summary plot for model explainability."""
    shap_path = PROJECT_ROOT / "models" / "shap_summary.png"
    if not shap_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Explainability artifact not found. Run training first.",
        )
    return FileResponse(shap_path)
    
    
@app.get("/store/{store_id}")
def get_store_metadata(store_id: int):
    """Fetches static store metadata (Type, Assortment, Distance) for UI pre-filling."""
    if STORE_DF is None:
        raise HTTPException(status_code=503, detail="Store metadata not loaded.")

    store_data = STORE_DF[STORE_DF["Store"] == store_id]
    if store_data.empty:
        raise HTTPException(status_code=404, detail=f"Store ID {store_id} not found.")

    row = store_data.iloc[0]
    return {
        "StoreType": str(row["StoreType"]),
        "Assortment": str(row["Assortment"]),
        "CompetitionDistance": float(row["CompetitionDistance"])
        if pd.notnull(row["CompetitionDistance"])
        else 0.0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Transaction-level sales prediction."""
    if MODEL is None or STORE_MEANS is None:
        raise HTTPException(
            status_code=503,
            detail="Model or target means not loaded. Ensure training has been run.",
        )

    # Data-poisoning defence: CompetitionDistance already bounded by schema (le=100_000).
    # Secondary guard for belt-and-suspenders safety on non-schema paths.
    if (
        request.CompetitionDistance is not None
        and request.CompetitionDistance > 100_000
    ):
        INFERENCE_ANOMALIES_BLOCKED.inc()
        logger.warning(
            "Anomalous CompetitionDistance=%.2f blocked for Store=%d.",
            request.CompetitionDistance,
            request.Store,
        )
        raise HTTPException(
            status_code=422,
            detail="CompetitionDistance exceeds plausible range (>100 000 m). Request blocked.",
        )

    try:
        # 1. Reconstruct input DataFrame
        input_data = pd.DataFrame([request.model_dump()])

        # 2. Enrich with store metadata (CompetitionDistance from store.csv if not provided)
        if STORE_DF is not None:
            enriched = pd.merge(
                input_data, STORE_DF, on="Store", how="left", suffixes=("", "_store")
            )
            if request.CompetitionDistance is not None:
                enriched["CompetitionDistance"] = request.CompetitionDistance
        else:
            enriched = input_data

        # 3. Stateless feature transforms (dates, OHE, LogCompDist, column alignment)
        # train_comp_median is not critical for a single row — GLOBAL_MEAN gives a
        # reasonable fallback; STORE_MEANS already captures store-level distance signal.
        train_comp_median = enriched["CompetitionDistance"].fillna(0).iloc[0]
        from rossmann_ops.features import build_features as _bf  # noqa: F401

        config_path = PROJECT_ROOT / "configs" / "params.yaml"
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)

        processed = build_features(
            enriched,
            train_comp_median=float(train_comp_median),
            expected_ohe_cols=params["features"]["ohe_expected_columns"],
        )

        # 4. Apply Store Target Encoding
        processed = apply_target_encoding(processed, STORE_MEANS, GLOBAL_MEAN)

        # 5. Align feature columns to training order
        feature_cols = [
            "DayOfWeek",
            "Promo",
            "Year",
            "Month",
            "WeekOfYear",
            "LogCompDist",
            "Store_TargetMean",
        ] + params["features"]["ohe_expected_columns"]

        for col in feature_cols:
            if col not in processed.columns:
                processed[col] = 0

        # 6. Inference — model predicts in log-space, we restore to real-space
        prediction_log = MODEL.predict(processed[feature_cols])
        prediction = float(np.expm1(prediction_log[0]))

        SALES_INFERENCE_TOTAL.inc()
        return {
            "Store": request.Store,
            "Date": request.Date,
            "PredictedSales": prediction,
            "ModelVersion": MODEL_VERSION,
        }

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from None


@app.post("/drift-trigger")
def drift_trigger(request: DriftRequest):
    """Stub for retraining trigger. Logs drift event for monitoring."""
    logger.info("Drift check triggered for %d records.", len(request.sales_data))
    return {
        "drift_detected": False,
        "timestamp": pd.Timestamp.now().isoformat(),
        "message": "Drift monitoring stub active.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
