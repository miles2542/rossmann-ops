import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rossmann_ops.data_validation import validate_data
from rossmann_ops.features import apply_target_encoding, build_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error. Ignores zero-sales rows."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.maximum(y_pred, 1.0)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def train_production_model() -> None:
    # 1. Load config
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "params.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    rs = config["train"]["random_state"]
    prod_cfg = config["model"]["production"]
    split_cfg = config["data_split"]
    ohe_cols = config["features"]["ohe_expected_columns"]

    # 2. Load and Validate Raw Data
    logger.info("Loading raw data...")
    train_df = pd.read_csv(project_root / config["data"]["raw_train"], low_memory=False)
    store_df = pd.read_csv(project_root / config["data"]["raw_store"])
    df = pd.merge(train_df, store_df, on="Store", how="left")
    df = validate_data(df)

    # 3. Filter: open days with sales > 0 only
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # 4. Chronological Sort & Strict Data Split
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True, ignore_index=True)

    max_date = df["Date"].max()
    sim_cutoff = max_date - pd.Timedelta(days=split_cfg["simulation_days"])
    holdout_cutoff = sim_cutoff - pd.Timedelta(days=split_cfg["holdout_days"])

    # Simulation set is never touched during training or evaluation
    df_cv = df[df["Date"] <= holdout_cutoff].copy()
    df_holdout = df[(df["Date"] > holdout_cutoff) & (df["Date"] <= sim_cutoff)].copy()

    logger.info(
        "Split complete. CV: %d rows, Holdout: %d rows, Sim set excluded: %d rows.",
        len(df_cv),
        len(df_holdout),
        len(df[df["Date"] > sim_cutoff]),
    )

    # 5. Compute train-set statistics for leak-free transformations
    train_comp_median = df_cv["CompetitionDistance"].median()
    logger.info("Train CompetitionDistance median: %.2f", train_comp_median)

    # 6. Apply Stateless Feature Engineering
    # Pass ohe_cols to both slices so column alignment is enforced regardless
    # of which holiday categories appear within each time window.
    df_cv = build_features(
        df_cv, train_comp_median=train_comp_median, expected_ohe_cols=ohe_cols
    )
    df_holdout = build_features(
        df_holdout, train_comp_median=train_comp_median, expected_ohe_cols=ohe_cols
    )

    # 7. Expanding Mean Target Encoding on df_cv only.
    # Expanding mean uses only past observations per store — zero leakage.
    df_cv["Store_TargetMean"] = df_cv.groupby("Store")["Sales"].transform(
        lambda x: x.shift().expanding().mean()
    )
    global_mean = float(df_cv["Sales"].mean())
    df_cv["Store_TargetMean"] = df_cv["Store_TargetMean"].fillna(global_mean)

    # Save final per-store means: one lookup value per store for inference
    final_store_means: dict[int, float] = (
        df_cv.groupby("Store")["Sales"].mean().to_dict()
    )
    # Apply to holdout using training-set means (no future peek)
    df_holdout = apply_target_encoding(df_holdout, final_store_means, global_mean)

    # 8. Define Feature Columns
    feature_cols = [
        "DayOfWeek",
        "Promo",
        "Year",
        "Month",
        "WeekOfYear",
        "LogCompDist",
        "Store_TargetMean",
    ] + ohe_cols

    X_cv = df_cv[feature_cols]
    y_cv_log = np.log1p(df_cv["Sales"])
    X_holdout = df_holdout[feature_cols]
    y_holdout = df_holdout["Sales"].to_numpy()

    # 9. Persist Target Encoding Artifact
    target_means_path = project_root / config["model"]["target_means_path"]
    target_means_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_means_path, "w") as f:
        json.dump({"store_means": final_store_means, "global_mean": global_mean}, f)
    logger.info("Store target means saved to %s.", target_means_path)

    # 10. MLflow Training Run
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{project_root}/mlruns/mlflow.db")
    )
    mlflow.set_experiment(config["train"]["experiment_name"])
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run() as run:
        # Log configuration
        mlflow.log_params(
            {
                "model_type": config["model"]["type"],
                "n_estimators": prod_cfg["n_estimators"],
                "max_depth": prod_cfg["max_depth"],
                "min_samples_split": prod_cfg["min_samples_split"],
                "random_state": rs,
                "holdout_days": split_cfg["holdout_days"],
                "simulation_days": split_cfg["simulation_days"],
                "n_cv_rows": len(X_cv),
                "train_comp_median": train_comp_median,
            }
        )

        logger.info("Starting Random Forest training...")
        model = RandomForestRegressor(
            n_estimators=prod_cfg["n_estimators"],
            max_depth=prod_cfg["max_depth"],
            min_samples_split=prod_cfg["min_samples_split"],
            random_state=rs,
            n_jobs=-1,
        )

        # Cast to float64 for consistent MLflow schema and API type-resilience
        X_cv_train = X_cv.astype(np.float64)
        model.fit(X_cv_train, y_cv_log)

        # 11. Holdout Evaluation in Real Sales Space
        logger.info("Evaluating on holdout set...")
        X_holdout_test = X_holdout.astype(np.float64)
        preds_log = model.predict(X_holdout_test)
        preds = np.expm1(preds_log)

        holdout_rmspe = rmspe(y_holdout, preds)
        holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, preds)))
        holdout_mae = float(mean_absolute_error(y_holdout, preds))
        holdout_r2 = float(r2_score(y_holdout, preds))

        mlflow.log_metrics(
            {
                "holdout_rmspe": holdout_rmspe,
                "holdout_rmse": holdout_rmse,
                "holdout_mae": holdout_mae,
                "holdout_r2": holdout_r2,
            }
        )
        logger.info(
            "Holdout: RMSPE=%.4f, RMSE=%.2f, MAE=%.2f, R2=%.4f",
            holdout_rmspe,
            holdout_rmse,
            holdout_mae,
            holdout_r2,
        )

        # 12. SHAP Explainability (sample for speed)
        logger.info("Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(model)
        sample_X = X_cv.sample(min(500, len(X_cv)), random_state=rs)
        shap_values = explainer.shap_values(sample_X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.tight_layout()

        shap_out_path = project_root / "models" / "shap_summary.png"
        shap_out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(shap_out_path)
        plt.close()

        with tempfile.TemporaryDirectory() as tmp_dir:
            shap_tmp = os.path.join(tmp_dir, "shap_summary.png")
            plt.savefig(shap_tmp) if False else shutil.copy(shap_out_path, shap_tmp)
            mlflow.log_artifact(shap_tmp)

        # 13. Log Model to MLflow Registry
        mlflow.sklearn.log_model(model, artifact_path="production_model")

        # 14. Save Model Locally for Docker/CI builds
        local_model_dir = project_root / config["model"]["save_path"]
        if local_model_dir.exists():
            shutil.rmtree(local_model_dir)
        mlflow.sklearn.save_model(model, str(local_model_dir))
        logger.info("Model saved to %s.", local_model_dir)

        logger.info("Training complete. Run ID: %s", run.info.run_id)


if __name__ == "__main__":
    train_production_model()
