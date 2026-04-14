import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def is_docker_running():
    try:
        # Cross-platform check
        result = subprocess.run(
            ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_docker_desktop():
    platform = sys.platform
    logger.info("Docker not running. Attempting to start Docker Desktop...")

    if platform == "win32":
        docker_path = Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe")
        if docker_path.exists():
            subprocess.Popen([str(docker_path)])
        else:
            logger.error("Docker Desktop executable not found at standard path.")
            sys.exit(1)

    elif platform == "darwin":
        # macOS
        subprocess.Popen(["open", "-a", "Docker"])

    else:
        # Standard Linux usually uses systemd
        logger.error(
            "Please start Docker manually (e.g., 'sudo systemctl start docker')."
        )
        sys.exit(1)


def main():
    if is_docker_running():
        logger.info("Docker is running.")
        return

    start_docker_desktop()

    timeout = 90
    elapsed = 0

    while elapsed < timeout:
        time.sleep(5)
        elapsed += 5
        logger.info(f"Waiting for Docker daemon... ({elapsed}/{timeout} s)")
        if is_docker_running():
            logger.info("Docker is now running.")
            return

    logger.error("Docker did not start in time. Please start it manually.")
    sys.exit(1)


if __name__ == "__main__":
    main()
