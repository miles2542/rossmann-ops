set shell := ["pwsh", "-c"]

setup:
    uv sync
    dvc pull

install:
    uv sync

lint *args:
    uv run ruff check . {{args}}

format:
    uv run ruff check . --select I --fix --unsafe-fixes
    uv run ruff format .

test:
    uv run pytest tests/ -v

train-prod:
    uv run python -m src.train_model

export-model:
    uv run python -m scripts.export_model

build-api: export-model
    docker build -f Dockerfile.api -t rossmann-api:latest .

build-ui:
    docker build -f Dockerfile.ui -t rossmann-ui:latest .
