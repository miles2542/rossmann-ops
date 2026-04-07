import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split

from src.data_validation import validate_data
from src.features import build_features

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_production_model() -> None:
    # 1. Setup paths and load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "params.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    # 2. Load and Validate Data
    logger.info("Loading raw data...")
    train_df = pd.read_csv(params["data"]["raw_train"], low_memory=False)
    store_df = pd.read_csv(params["data"]["raw_store"])
    df = pd.merge(train_df, store_df, on="Store", how="left")

    # Initial validation
    df = validate_data(df)

    # 3. Feature Engineering
    # Calculate competition fill value once from training distribution
    competition_fill = df["CompetitionDistance"].max() * 2
    logger.info(f"Using competition_distance_fill: {competition_fill}")

    # Build features
    df = build_features(df, fill_value=competition_fill)

    # 4. Filter and Split
    # Production model only trained on open days with sales > 0
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Define features (Maianh's expanded feature set)
    # We include our new time features and handle categorical encoding via XGBoost
    features = [
        "Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "WeekOfYear",
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"
    ]
    target = params["features"]["target"]

    X = df[features]
    y = df[target]

    # Handle basic categorical types for XGBoost if needed (StateHoliday, StoreType, etc.)
    # For simplicity in this step, we use numerical ones + Store
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
    )

    # 5. MLflow Tracking & Training
    mlflow.set_experiment("Rossmann_Production")
    mlflow.xgboost.autolog()

    with mlflow.start_run() as run:
        # Log custom parameter for fill value
        mlflow.log_param("competition_distance_fill", competition_fill)

        logger.info("Starting XGBoost training...")
        # Simple XGBoost regressor for production iteration
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=params["train"]["random_state"],
            tree_method="hist"  # for performance
        )

        model.fit(X_train, y_train)

        # 6. Explainability (SHAP)
        logger.info("Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(model)
        # Sample for speed in logging
        sample_X = X_test.sample(min(1000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample_X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.tight_layout()
        
        # Use system's native temporary directory for absolute robustness (CI/CD safe)
        with tempfile.TemporaryDirectory() as tmp_dir:
            shap_tmp_path = os.path.join(tmp_dir, "shap_summary.png")
            plt.savefig(shap_tmp_path)
            mlflow.log_artifact(shap_tmp_path)
            
        plt.close()

        # 7. Explicit Model Logging
        mlflow.xgboost.log_model(model, "production_model")

        logger.info(f"Production training complete. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_production_model()
