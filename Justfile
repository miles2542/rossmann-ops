set windows-shell := ["pwsh", "-c"]
export PYTHONPATH := "src"

# Ensures Docker Desktop is running before any build step.
# Attempts auto-launch on Windows; waits up to 90s for daemon to become ready.
check-docker:
	@uv run python scripts/check_docker.py

setup:
	uv sync
	dvc pull -f

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

build-api: check-docker export-model
	docker build -f Dockerfile.api -t rossmann-api:latest .

build-ui: check-docker
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
	@uv run python scripts/clean_artifacts.py

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

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

k8s-update:
	just build-api
	just build-ui
	just k8s-load
	-kubectl rollout restart deployment rossmann-api rossmann-ui
