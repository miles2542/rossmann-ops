import argparse
import os
from pathlib import Path


def _clean_dir(target: Path) -> None:
    """Wipe non-.dvc files from a directory and remove empty subdirs."""
    print(f"Cleaning directory: {target.name}/ (preserving .dvc files)")
    for path in target.rglob("*"):
        if path.is_file() and path.suffix != ".dvc":
            try:
                path.unlink()
            except Exception as e:
                print(f"  Failed delete {path}: {e}")
    for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if not args.yes:
        ans = input(
            "Are you sure you want to delete mlruns/, mlartifacts/, models/, mlflow.db? (Y/N) "
        )
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    for d in ["mlruns", "mlartifacts", "models"]:
        target = project_root / d
        if target.exists() and target.is_dir():
            _clean_dir(target)

    db = project_root / "mlflow.db"
    if db.exists():
        os.remove(db)
        print("Deleted root-level file: mlflow.db")

    print("Wiped artifacts successfully.")


if __name__ == "__main__":
    main()
