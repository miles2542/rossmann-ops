import logging
import random
import time
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load config
_project_root = Path(__file__).resolve().parents[1]
_config_path = _project_root / "configs" / "observability.yaml"
with open(_config_path, "r") as _f:
    _cfg = yaml.safe_load(_f)

API_URL = _cfg["demo"]["api_url"]
DASHBOARD_URL = _cfg["demo"]["dashboard_url"]
P1 = _cfg["demo"]["phases"]["normal"]
P2 = _cfg["demo"]["phases"]["schema_error"]
P3 = _cfg["demo"]["phases"]["attack"]


def get_base_payload():
    return {
        "Store": 1,
        "DayOfWeek": 1,
        "Date": "2025-01-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
        "StoreType": "c",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "CompetitionOpenSinceMonth": 9.0,
        "CompetitionOpenSinceYear": 2008.0,
        "Promo2": 0,
        "Promo2SinceWeek": 0.0,
        "Promo2SinceYear": 0.0,
        "PromoInterval": "0",
    }


def run_demo():
    print("\n🚀 ROSSMANN OBSERVABILITY LIVE DEMO 🚀")
    print("========================================")
    print(f"Targeting API: {API_URL}")
    print(f"Ensure Grafana is open at {DASHBOARD_URL}")
    print("Standard Refresh: 5s | Time Window: Last 5m\n")

    # PHASE 1: NORMAL TRAFFIC
    print("🟢 PHASE 1: Sending Regular Production Traffic...")
    print("Goal: Watch 'Global RPS' and 'Total Predictions' climb. Graph stays Green.")
    for i in range(P1["requests"]):
        payload = get_base_payload()
        payload["Store"] = random.randint(1, 1000)
        try:
            requests.post(API_URL, json=payload)
            if i % 50 == 0:
                print(f"  Sent {i + 50}/{P1['requests']} normal requests...")
        except Exception:
            pass
        time.sleep(P1["sleep_interval_sec"])

    print("\n🟡 PHASE 2: Simulating Malformed/Schema Errors...")
    print("Goal: WatchPie Chart slice 'HTTP 4xx' appear and Error Gauge flicker.")
    for i in range(P2["requests"]):
        bad_payload = {"Store": "INVALID_TYPE", "Missing": "Fields"}
        try:
            requests.post(API_URL, json=bad_payload)
            if i % 25 == 0:
                print(f"  Sent {i + 25}/{P2['requests']} malformed requests...")
        except Exception:
            pass
        time.sleep(P2["sleep_interval_sec"])

    print("\n🔴 PHASE 3: Launching Poisoning Attack (Anomaly Guard)...")
    print("Goal: 'Anomalies Blocked' will tick up. Error Rate Gauge hits Red (20%+).")
    for i in range(P3["requests"]):
        poisoned_payload = get_base_payload()
        # Trips the guard added in main.py (CompetitionDistance > 50000)
        poisoned_payload["CompetitionDistance"] = 999999.0
        try:
            requests.post(API_URL, json=poisoned_payload)
            if i % 30 == 0:
                print(f"  Sent {i + 30}/{P3['requests']} anomalies...")
        except Exception:
            pass
        time.sleep(P3["sleep_interval_sec"])

    print("\n✅ DEMO COMPLETE")
    print("Check your dashboard for the final status distribution.")


if __name__ == "__main__":
    run_demo()
