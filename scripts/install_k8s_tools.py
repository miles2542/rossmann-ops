"""Cross-platform installer for kind (Kubernetes IN Docker) and kubectl.

Invoked by:
    just install-k8s-tools

Supports Windows (winget), macOS (brew), and Linux (binary download).
Skips any tool already present on PATH.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

KIND_VERSION = "v0.31.0"


def _machine() -> str:
    """Normalise platform.machine() to the kind/kubectl arch identifier."""
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _present(tool: str) -> bool:
    return shutil.which(tool) is not None


# ── Platform installers ────────────────────────────────────────────────────

def install_windows(missing: list[str]) -> None:
    if not _present("winget"):
        print(
            "ERROR: winget not found.\n"
            "Install the App Installer from the Microsoft Store: https://aka.ms/getwinget"
        )
        sys.exit(1)

    winget_ids = {"kind": "Kubernetes.kind", "kubectl": "Kubernetes.kubectl"}
    for tool in missing:
        _run(["winget", "install", "--id", winget_ids[tool], "--exact", "--source", "winget"])


def install_macos(missing: list[str]) -> None:
    if not _present("brew"):
        print(
            "ERROR: Homebrew not found.\n"
            "Install from https://brew.sh and re-run."
        )
        sys.exit(1)

    _run(["brew", "install", *missing])


def install_linux(missing: list[str]) -> None:
    arch = _machine()
    if arch not in ("amd64", "arm64"):
        print(f"WARNING: Untested architecture '{arch}'. Proceeding anyway.")

    for tool in missing:
        if tool == "kind":
            url = f"https://kind.sigs.k8s.io/dl/{KIND_VERSION}/kind-linux-{arch}"
            dest = Path("/tmp/kind")
            print(f"  Downloading kind {KIND_VERSION} ({arch}) from {url} ...")
            urllib.request.urlretrieve(url, dest)
            dest.chmod(0o755)
            _run(["sudo", "mv", str(dest), "/usr/local/bin/kind"])
            print("  kind installed -> /usr/local/bin/kind")

        elif tool == "kubectl":
            ver_url = "https://dl.k8s.io/release/stable.txt"
            with urllib.request.urlopen(ver_url) as resp:
                k_ver = resp.read().decode().strip()
            url = f"https://dl.k8s.io/release/{k_ver}/bin/linux/{arch}/kubectl"
            dest = Path("/tmp/kubectl")
            print(f"  Downloading kubectl {k_ver} ({arch}) from {url} ...")
            urllib.request.urlretrieve(url, dest)
            dest.chmod(0o755)
            _run(["sudo", "mv", str(dest), "/usr/local/bin/kubectl"])
            print("  kubectl installed -> /usr/local/bin/kubectl")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    missing = [t for t in ("kind", "kubectl") if not _present(t)]

    if not missing:
        print("kind and kubectl are already on PATH. Nothing to do.")
        return

    print(f"Installing: {', '.join(missing)}")

    system = platform.system()
    if system == "Windows":
        install_windows(missing)
    elif system == "Darwin":
        install_macos(missing)
    elif system == "Linux":
        install_linux(missing)
    else:
        print(f"Unsupported OS: {system}")
        print("Install manually:")
        print("  kind:    https://kind.sigs.k8s.io/docs/user/quick-start/")
        print("  kubectl: https://kubernetes.io/docs/tasks/tools/")
        sys.exit(1)

    print("\nDone. Restart your shell (or reopen VS Code) to refresh PATH.")


if __name__ == "__main__":
    main()
