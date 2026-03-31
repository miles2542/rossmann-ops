import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_validation import validate_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_baseline() -> None:
    # 1. Use absolute path relative to the script location
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "params.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    # 2. Load and Validate Data
    train_df = pd.read_csv(params["data"]["raw_train"], low_memory=False)
    store_df = pd.read_csv(params["data"]["raw_store"])
    df = pd.merge(train_df, store_df, on="Store", how="left")

    df = validate_data(df)

    # 3. Preprocessing & Filtering
    # Filter for open stores with sales > 0 to normalize baseline variance
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    features = params["features"]["baseline"]
    target = params["features"]["target"]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
    )

    # 4. Pipeline Construction
    categorical_features = params["features"]["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=params["model"]["alpha"])),
        ]
    )

    # 5. MLflow Tracking
    mlflow.set_experiment(params["train"]["experiment_name"])
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        logger.info("Starting baseline model training (Ridge)...")
        model_pipeline.fit(X_train, y_train)

        # Predictions & Metrics
        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Explicit logging for high visibility in UI
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        logger.info(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # 6. Serialization
        save_path = Path(params["model"]["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_pipeline, save_path)
        logger.info(f"Model serialized to {save_path}")


if __name__ == "__main__":
    train_baseline()
