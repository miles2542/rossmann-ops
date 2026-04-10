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

setup-prod:
    just setup
    just train-prod
    just build-api
    just build-ui
    just k8s-up
    just k8s-load
    just k8s-deploy

clean-artifacts:
    @$confirm = Read-Host 'Are you sure you want to delete mlruns/ and models/ and mlflow.db? (Y/N)'; if ($confirm -eq 'Y') { Remove-Item -Recurse -Force mlruns, mlartifacts, models, mlflow.db -ErrorAction SilentlyContinue; Write-Host 'Wiped.' }

k8s-up:
    @if (-not (kind get clusters | Select-String 'rossmann-cluster')) { kind create cluster --name rossmann-cluster --config k8s/kind-config.yaml } else { Write-Host 'Cluster already exists, skipping creation.' }

k8s-down:
    kind delete cluster --name rossmann-cluster

k8s-load:
    kind load docker-image rossmann-api:latest rossmann-ui:latest --name rossmann-cluster

k8s-deploy:
    kubectl apply -f k8s/api.yaml
    kubectl apply -f k8s/ui.yaml

k8s-status:
    kubectl get pods,services

mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
