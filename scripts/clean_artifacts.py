import os
import shutil
from pathlib import Path


def main():
    # Dynamically resolve project root regardless of where script is called
    project_root = Path(__file__).resolve().parents[1]

    ans = input(
        "Are you sure you want to delete mlruns/, mlartifacts/, models/, mlflow.db? (Y/N) "
    )
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    dirs_to_delete = ["mlruns", "mlartifacts", "models"]
    files_to_delete = ["mlflow.db"]

    for d in dirs_to_delete:
        target = project_root / d
        if target.exists() and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"Deleted directory: {d}/")

    for f in files_to_delete:
        target = project_root / f
        if target.exists() and target.is_file():
            os.remove(target)
            print(f"Deleted file: {f}")

    print("Wiped artifacts successfully.")


if __name__ == "__main__":
    main()
