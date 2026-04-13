import logging
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_latest_model():
    """
    Finds the latest run in 'Rossmann_Production' or 'Rossmann_Baseline'
    and copies the model artifact to models/production_model.pkl.
    """
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    )
    client = MlflowClient()
    experiment_name = "Rossmann_Production"

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error(
            f"Experiment '{experiment_name}' not found. "
            "Please run 'just train-prod' or 'dvc pull' first."
        )
        return

    # 2. Search for latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    if not runs:
        logger.error(f"No runs found in experiment '{experiment_name}'.")
        return

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    logger.info(f"Found latest run: {run_id} (Created: {latest_run.info.start_time})")

    # 3. Path setup
    project_root = Path(__file__).parent.parent

    # MLflow logs models as a full directory (MLmodel, conda.yaml, model binary).
    # Exporting the entire directory is the most robust practice across all platforms.
    artifact_path = "production_model"

    # Download the entire directory structure
    local_path = client.download_artifacts(run_id, artifact_path)

    dest_path = project_root / "models" / "production_model"

    # Clean up old directory if it exists to prevent stale files mixing
    if dest_path.exists():
        shutil.rmtree(dest_path)

    shutil.copytree(local_path, dest_path)
    logger.info(f"Successfully exported full model directory to {dest_path}")

    # 4. Export SHAP summary if exists
    try:
        shap_path = client.download_artifacts(run_id, "shap_summary.png")
        if shap_path:
            shutil.copy(shap_path, project_root / "models" / "shap_summary.png")
            logger.info("Exported SHAP summary to models/shap_summary.png")
    except Exception as e:
        logger.warning(f"Could not export SHAP summary: {e}")


if __name__ == "__main__":
    export_latest_model()
