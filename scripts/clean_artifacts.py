import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Dynamically resolve project root regardless of where script is called
    project_root = Path(__file__).resolve().parents[1]

    if not args.yes:
        ans = input(
            "Are you sure you want to delete mlruns/, mlartifacts/, models/, mlflow.db? (Y/N) "
        )
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    dirs_to_process = ["mlruns", "mlartifacts", "models"]
    files_to_delete = ["mlflow.db"] # Specifically target this outside if needed

    for d in dirs_to_process:
        target = project_root / d
        if target.exists() and target.is_dir():
            print(f"Cleaning directory: {d}/ (preserving .dvc files)")
            # Recursive cleanup
            for path in target.rglob("*"):
                if path.is_file() and path.suffix != ".dvc":
                    try:
                        path.unlink()
                        # print(f"  Deleted: {path.relative_to(project_root)}")
                    except Exception as e:
                        print(f"  Failed delete {path}: {e}")
            
            # Clean up empty subdirectories (except DVC metadata folders if any)
            # We walk bottom-up to remove childless dirs
            for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()

    for f in files_to_delete:
        target = project_root / f
        if target.exists() and target.is_file():
            os.remove(target)
            print(f"Deleted root-level file: {f}")

    print("Wiped artifacts successfully.")


if __name__ == "__main__":
    main()
