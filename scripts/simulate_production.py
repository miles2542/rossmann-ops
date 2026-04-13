import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from scipy import stats
from sklearn.model_selection import train_test_split

# Auto-resolve PYTHONPATH so we don't need env vars in terminal
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parents[1]
_config_path = _project_root / "configs" / "params.yaml"
with open(_config_path, "r") as _f:
    _cfg = yaml.safe_load(_f)

_sim = _cfg["pipeline"]["simulation"]
_gh = _cfg["pipeline"]["github"]

API_URL = os.getenv("API_URL", _sim["api_url"])
GITHUB_PAT = os.getenv("GITHUB_PAT")
REPO_OWNER = _gh["owner"]
REPO_NAME = _gh["repo"]
Z_SCORE_THRESHOLD = _sim["z_score_threshold"]
P_VALUE_THRESHOLD = _sim["p_value_threshold"]
DRIFT_FEATURE = _sim["drift_feature"]
DRIFT_SHIFT = _sim["drift_shift"]


def get_test_data():
    """Reproduces the exact train/test split from the training pipeline."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "params.yaml"
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    logger.info("Loading dataset to extract baseline test set...")
    train_df = pd.read_csv(project_root / params["data"]["raw_train"], low_memory=False)
    store_df = pd.read_csv(project_root / params["data"]["raw_store"])

    df = pd.merge(train_df, store_df, on="Store", how="left")

    # Matching the exact filtering applied in train_model.py
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    from rossmann_ops.features import build_features

    competition_fill = df["CompetitionDistance"].max() * 2
    df = build_features(df, fill_value=competition_fill)

    features = [
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
    target = params["features"]["target"]

    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
    )
    return X_test, y_test


def simulate_schema_error():
    """Sends malformed JSON to trigger API 422 validation limits."""
    logger.info("--- Mode: Schema Validation Check ---")
    bad_payload = {
        "Store": 1,
        "DayOfWeek": "NOT_AN_INTEGER",  # Wrong type
        "Date": "2025-01-01",
        # Missing "Promo" and "StateHoliday"
    }
    logger.info(f"Sending malformed payload: {bad_payload}")
    try:
        response = requests.post(f"{API_URL}/predict", json=bad_payload)
        logger.info(f"API Response Code: {response.status_code}")
        if response.status_code == 422:
            logger.info(
                "Success! API blocked malformed schema (422 Unprocessable Entity - Pydantic/Validation Error)."
            )
            logger.info(f"Details: {response.json()}")
        else:
            logger.warning(
                "Failed to trigger 422 validation. The API might have accepted it or failed differently."
            )
    except Exception as e:
        logger.error(f"Failed to connect to API: {e}")


def simulate_attack():
    """Simulates a batch data poisoning attempt and locally computes Z-Score before aborting."""
    logger.info("--- Mode: Layered Defense / Attack Handling ---")
    _, y_test = get_test_data()

    # Simulate an incoming batch with injected malicious extreme outliers
    malicious_batch = y_test.copy().values
    malicious_batch[0:10] = 999999999  # Astronomically high Sales values

    logger.info("Calculating Z-scores locally to defend against data poisoning...")
    # Z-Score = (X - mean) / std
    mean_val = np.mean(malicious_batch)
    std_val = np.std(malicious_batch)
    z_scores = (malicious_batch - mean_val) / std_val
    max_z = np.max(z_scores)

    logger.info(f"Maximum detected Z-Score in batch: {max_z:.2f}")
    if max_z > Z_SCORE_THRESHOLD:
        logger.error(
            f"Anomaly/Poisoning detected! Z-Score {max_z:.2f} exceeds threshold ({Z_SCORE_THRESHOLD}). Aborting ingestion."
        )
        sys.exit(1)
    else:
        logger.info("Batch looks normal. Proceeding...")


def trigger_github_workflow():
    if not GITHUB_PAT:
        logger.warning(
            "GITHUB_PAT environment variable not set. Cannot trigger GitHub Actions repository dispatch."
        )
        return

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_PAT}",
    }
    payload = {"event_type": "drift_detected"}

    logger.info("Triggering GitHub Actions workflow via repository_dispatch...")
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 204:
        logger.info("Successfully dispatched 'drift_detected' event to GitHub!")
    else:
        logger.error(
            f"Failed to dispatch event. Code: {res.status_code}. Response: {res.text}"
        )


def simulate_drift():
    """Checks for statistical drift via KS-Test, triggers GitHub action on failure."""
    logger.info("--- Mode: Continuous Training / Drift Simulation ---")
    X_test, _ = get_test_data()

    baseline_feature = X_test[DRIFT_FEATURE].fillna(0).values

    # Simulate feature drift by shifting the distribution
    shifted_feature = baseline_feature + DRIFT_SHIFT

    logger.info("Running Kolmogorov-Smirnov (KS-Test) for drift detection...")
    stat, p_value = stats.ks_2samp(baseline_feature, shifted_feature)

    logger.info(f"KS Statistic: {stat:.4f}, p-value: {p_value:.4e}")

    if p_value < P_VALUE_THRESHOLD:
        logger.warning(f"Statistical Data Drift Detected (p-value {p_value:.4e} < {P_VALUE_THRESHOLD})!")
        trigger_github_workflow()
    else:
        logger.info("No significant drift detected. Model represents real-world well.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["schema", "attack", "drift"],
        required=True,
        help="Simulation mode: schema (API malformed data), attack (Poisoning), drift (CT Trigger).",
    )

    args = parser.parse_args()

    if args.mode == "schema":
        simulate_schema_error()
    elif args.mode == "attack":
        simulate_attack()
    elif args.mode == "drift":
        simulate_drift()
