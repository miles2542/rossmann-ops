set shell := ["pwsh", "-c"]

install:
    uv sync

lint *args:
    uv run ruff check . {{args}}

format:
    uv run ruff check . --select I --fix --unsafe-fixes
    uv run ruff format .

test:
    uv run pytest tests/ -v
