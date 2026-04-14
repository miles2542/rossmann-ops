import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parents[1]
    kind_config_path = project_root / "k8s" / "kind-config.yaml"
    cluster_name = "rossmann-cluster"

    try:
        result = subprocess.run(
            ["kind", "get", "clusters"], capture_output=True, text=True, check=True
        )

        if cluster_name in result.stdout.splitlines():
            logger.info("Cluster 'rossmann-cluster' already exists, skipping creation.")
            return

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check kind clusters: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Error: 'kind' is not installed or not in PATH.")
        sys.exit(1)

    # Need to create it
    logger.info("Creating cluster 'rossmann-cluster'...")
    try:
        subprocess.run(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                cluster_name,
                "--config",
                str(kind_config_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create cluster: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
