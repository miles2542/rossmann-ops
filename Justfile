set windows-shell := ["pwsh", "-c"]
set dotenv-load := true
export PYTHONPATH := "src"
export PYTHONUTF8 := "1"
export MLFLOW_TRACKING_URI := env_var("MLFLOW_TRACKING_URI")
export MLFLOW_TRACKING_USERNAME := env_var("MLFLOW_TRACKING_USERNAME")
export MLFLOW_TRACKING_PASSWORD := env_var("MLFLOW_TRACKING_PASSWORD")

# Ensures Docker Desktop is running before any build step.
# Attempts auto-launch on Windows; waits up to 90s for daemon to become ready.
check-docker:
	@uv run python scripts/check_docker.py

setup:
	uv sync
	just pull

install:
	uv sync

pull:
	@uv run dvc remote modify --local dagshub user {{env_var("DAGSHUB_USERNAME")}}
	@uv run dvc remote modify --local dagshub password {{env_var("DAGSHUB_PAT")}}
	uv run dvc pull

push:
	@uv run dvc remote modify --local dagshub user {{env_var("DAGSHUB_USERNAME")}}
	@uv run dvc remote modify --local dagshub password {{env_var("DAGSHUB_PAT")}}
	uv run dvc push

lint *args:
	uv run ruff check . {{args}}

format:
	uv run ruff check . --select I --fix --unsafe-fixes
	uv run ruff format .

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

train-prod:
	uv run python -m rossmann_ops.train_model

export-model:
	uv run python -m scripts.export_model

build-api: check-docker
	docker build -f Dockerfile.api -t rossmann-api:latest .

build-ui: check-docker
	docker build -f Dockerfile.ui -t rossmann-ui:latest .

setup-prod:
	just setup
	@echo "Cleaning old MLflow/DVC artifacts to avoid absolute path issues..."
	just clean-artifacts --yes
	just train-prod
	just build-api
	just build-ui
	just k8s-up
	just k8s-load
	just k8s-deploy

clean-artifacts *args:
	@uv run python scripts/clean_artifacts.py {{args}}

k8s-up:
	@uv run python scripts/k8s_up.py

k8s-down:
	kind delete cluster --name rossmann-cluster

k8s-load:
	kind load docker-image rossmann-api:latest rossmann-ui:latest --name rossmann-cluster

k8s-deploy:
	kubectl apply -f k8s/api.yaml
	kubectl apply -f k8s/ui.yaml

k8s-status:
	kubectl get pods,services,servicemonitors -A

k8s-monitoring-setup:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install prom -n monitoring --create-namespace \
		prometheus-community/kube-prometheus-stack \
		--set grafana.service.type=NodePort \
		--set grafana.service.nodePort=30200 \
		--set grafana.adminPassword=prom-operator \
		--set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
		--set prometheus.prometheusSpec.scrapeInterval=1s \
		--set prometheus.service.type=NodePort \
		--set prometheus.service.nodePort=30300

k8s-apply-monitors:
	kubectl apply -f k8s/servicemonitor.yaml
	kubectl apply -f k8s/grafana-dashboard.yaml

deploy-all: \
	setup-prod \
	k8s-monitoring-setup \
	k8s-apply-monitors
	@echo ""
	@echo "  Full stack deployed. Allow 30-60s for pods to stabilize."
	@echo ""
	@echo "  Open in browser:"
	@echo "    Streamlit UI  ->  http://localhost:30000"
	@echo "    API Docs      ->  http://localhost:30100/docs"
	@echo "    API Health    ->  http://localhost:30100/health"
	@echo "    Grafana       ->  http://localhost:30200  (admin / prom-operator)"
	@echo "    Prometheus    ->  http://localhost:30300"
	@echo ""

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

k8s-update:
	just build-api
	just build-ui
	just k8s-load
	-kubectl rollout restart deployment rossmann-api rossmann-ui

# ── Local development servers (no Docker, no K8s) ──────────────────────────

serve-api:
	@echo ""
	@echo "  Starting FastAPI inference server ..."
	@echo "  Once running, open:  http://localhost:8000/docs"
	@echo ""
	uv run uvicorn rossmann_ops.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-ui:
	@echo ""
	@echo "  Starting Streamlit dashboard ..."
	@echo "  Once running, open:  http://localhost:8501"
	@echo ""
	uv run streamlit run ui/app.py

demo:
	@echo ""
	@echo "  Running observability demo (3 phases: normal / schema errors / poisoning attack)."
	@echo "  Requires inference API running on port 30100 (K8s or Docker Compose)."
	@echo "  Watch live Grafana metrics at:  http://localhost:30200  (K8s only)"
	@echo ""
	uv run python scripts/observability_demo.py

# ── Docker Compose — published images, no K8s required ─────────────────────

docker-up: check-docker
	docker compose pull
	docker compose up -d
	@echo ""
	@echo "  Services started from DockerHub images. Open:"
	@echo "    Streamlit UI  ->  http://localhost:30000"
	@echo "    API Docs      ->  http://localhost:30100/docs"
	@echo "    API Health    ->  http://localhost:30100/health"
	@echo ""
	@echo "  Note: Grafana/Prometheus not available in Docker Compose mode."
	@echo "  Run 'just deploy-all' for the full K8s stack with observability."
	@echo ""

docker-down:
	docker compose down

# ── Tool installation ───────────────────────────────────────────────────────

install-k8s-tools:
	uv run python scripts/install_k8s_tools.py
