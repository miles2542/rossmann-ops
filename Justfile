set windows-shell := ["pwsh", "-c"]
export PYTHONPATH := "src"

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
    uv run python -m rossmann_ops.train_model

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
    @uv run python -c "import shutil, os; x = input('Are you sure you want to delete mlruns/, models/, mlflow.db? (Y/N) '); [(shutil.rmtree(d, ignore_errors=True) if os.path.exists(d) else None) for d in ['mlruns', 'mlartifacts', 'models']] if x.lower()=='y' else None; (os.remove('mlflow.db') if os.path.exists('mlflow.db') else None) if x.lower()=='y' else None; print('Wiped.') if x.lower()=='y' else print('Aborted.')"

k8s-up:
    @uv run python -c "import subprocess as sp; exists = b'rossmann-cluster' in sp.run(['kind', 'get', 'clusters'], capture_output=True).stdout; sp.run(['kind', 'create', 'cluster', '--name', 'rossmann-cluster', '--config', 'k8s/kind-config.yaml']) if not exists else print('Cluster already exists, skipping creation.')"

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
