<div align="center">
    
# Rossmann Store Sales Demand Forecasting (MLOps Pipeline)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=apache)](LICENSE)
[![CI Pipeline](https://github.com/miles2542/rossmann-ops/actions/workflows/mlops_pipeline.yaml/badge.svg)](https://github.com/miles2542/rossmann-ops/actions/workflows/mlops_pipeline.yaml)
[![Test Coverage](https://img.shields.io/badge/Coverage-52%25-yellowgreen)](https://github.com/miles2542/rossmann-ops/tree/main/tests)

[![DVC](https://img.shields.io/badge/DVC-Data%20Versioned-945DD6?logo=dvc)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployed-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io)

![Grafana Monitoring Dashboard](docs/assets/grafana_demo.png)

*Predicts daily store sales for 1,115 Rossmann stores using a Random Forest model trained on 2+ years of transactional data. Deployed as a multi-replica microservice on a local Kubernetes cluster, with end-to-end automation covering CI, CD, and CT.*

</div>

---

## System Highlights

| Capability              | Implementation                                                                                                                    |
| :---------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| **Serving**             | FastAPI + 2-replica K8s deployment with `RollingUpdate` zero-downtime strategy                                                    |
| **Security**            | Layered defense: Pandera schema validation → Pydantic bounds → CompetitionDistance guard                                          |
| **Telemetry**           | Prometheus custom metrics (`sales_inference_total`, `inference_anomalies_blocked`) scraped at 1s intervals, visualized in Grafana |
| **Explainability**      | SHAP feature importance served via API, rendered live in the Streamlit dashboard                                                  |
| **Reproducibility**     | `uv.lock` + DVC-pinned artifacts; `just setup` installs everything deterministically                                              |
| **Continuous Training** | KS-Test drift detection triggers `repository_dispatch` → auto-retrain pipeline                                                    |

---

## Architecture

### Diagram 1 — Deployment Topology

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#0f172a", "primaryColor": "#1e293b", "primaryTextColor": "#f1f5f9", "primaryBorderColor": "#334155", "lineColor": "#64748b", "edgeLabelBackground": "#1e293b", "clusterBkg": "#111827", "clusterBorder": "#1f2937", "titleColor": "#f1f5f9"}}}%%
flowchart TB
    classDef user     fill:#1e3a5f,stroke:#60a5fa,color:#bfdbfe,stroke-width:2px
    classDef extStore fill:#2e1065,stroke:#a78bfa,color:#ede9fe,stroke-width:2px
    classDef port     fill:#1c1917,stroke:#44403c,color:#a8a29e,stroke-width:1px
    classDef svc      fill:#052e16,stroke:#22c55e,color:#bbf7d0,stroke-width:2px,font-weight:bold
    classDef pod      fill:#022c22,stroke:#16a34a,color:#86efac,stroke-width:1px
    classDef mon      fill:#431407,stroke:#f97316,color:#fdba74,stroke-width:2px
    classDef svcmon   fill:#1c1917,stroke:#fbbf24,color:#fde68a,stroke-width:1.5px

    User(["User / API Client"]):::user

    subgraph External ["  External Services  "]
        direction LR
        DagsHub[("DagsHub\n─────────────────\nDVC Remote Storage\nMLflow Experiment Tracking")]:::extStore
        DockerHub[("DockerHub\n─────────────────\nImage Registry\nmiles25420/rossmann-api\nmiles25420/rossmann-ui")]:::extStore
    end

    subgraph Cluster ["  Kubernetes Cluster — rossmann-cluster  |  KinD  |  1 Control Plane + 2 Workers  "]
        direction TB

        subgraph Ports ["  Host → Cluster Port Mappings  "]
            direction LR
            P30000>":30000  →  Streamlit UI"]:::port
            P30100>":30100  →  Inference API"]:::port
            P30200>":30200  →  Grafana"]:::port
            P30300>":30300  →  Prometheus"]:::port
        end

        subgraph App ["  Application Layer  —  default namespace  "]
            direction TB
            UISvc["Service: ui-service\nNodePort 30000"]:::svc
            APISvc["Service: api-service\nNodePort 30100"]:::svc

            subgraph UIPods ["  Streamlit UI Pods  (×2 replicas  ·  RollingUpdate  ·  maxUnavailable=0)  "]
                direction LR
                UI1[["Pod 1  ·  Streamlit :8501\nReadiness → GET /_stcore/health"]]:::pod
                UI2[["Pod 2  ·  Streamlit :8501\nReadiness → GET /_stcore/health"]]:::pod
            end

            subgraph APIPods ["  Inference API Pods  (×2 replicas  ·  RollingUpdate  ·  maxUnavailable=0)  "]
                direction LR
                API1[["Pod 1  ·  FastAPI :8000\nReadiness → GET /health\n/predict  /health  /store/{id}\n/health/shap  /metrics  /docs"]]:::pod
                API2[["Pod 2  ·  FastAPI :8000\nReadiness → GET /health\n/predict  /health  /store/{id}\n/health/shap  /metrics  /docs"]]:::pod
            end

            UISvc --> UI1 & UI2
            APISvc --> API1 & API2
        end

        subgraph Mon ["  Monitoring Stack  —  monitoring namespace  |  kube-prometheus-stack via Helm  "]
            direction LR
            SvcMon{{"ServiceMonitor\nlabel: app=rossmann-api\nscrapeInterval: 1s"}}:::svcmon
            Prom[("Prometheus\n:9090")]:::mon
            Graf["Grafana  :3000\nDashboard-as-Code\nConfigMap auto-provisioned"]:::mon
        end
    end

    User --> P30000 & P30100 & P30200 & P30300

    P30000 --> UISvc
    P30100 --> APISvc
    P30200 --> Graf
    P30300 --> Prom

    UI1 & UI2 -->|"K8s internal DNS\nhttp://api-service:8000"| APISvc
    SvcMon -->|"label-select: app=rossmann-api"| APISvc
    Prom -->|"scrape /metrics via ServiceMonitor"| SvcMon
    Graf -->|"PromQL queries"| Prom

    API1 & API2 -->|"load artifacts on startup\nlocal path or MLflow URI"| DagsHub
    DockerHub -->|"image pulled on pod scheduling"| API1 & API2 & UI1 & UI2
```

### Active Cluster State

![K8s Resource Status](docs/assets/k8s_status.png)

### Diagram 2 — Automated Pipeline  (CI / CD / CT)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#0f172a", "primaryColor": "#1e293b", "primaryTextColor": "#f1f5f9", "primaryBorderColor": "#334155", "lineColor": "#64748b", "edgeLabelBackground": "#1e293b", "clusterBkg": "#111827", "clusterBorder": "#1f2937"}}}%%
flowchart TD
    classDef trigger  fill:#1e1b4b,stroke:#818cf8,color:#e0e7ff,stroke-width:2px
    classDef job      fill:#0c2340,stroke:#3b82f6,color:#bfdbfe,stroke-width:2px,font-weight:bold
    classDef ctNode   fill:#2d0a47,stroke:#c084fc,color:#f3e8ff,stroke-width:2px
    classDef artifact fill:#042f2e,stroke:#14b8a6,color:#99f6e4,stroke-width:2px

    subgraph Triggers ["  Pipeline Triggers  "]
        direction LR
        T1{{"Push\nany branch"}}:::trigger
        T2{{"Push\nmain or tag"}}:::trigger
        T3{{"Push\nversion tag  v*.*.*"}}:::trigger
        T4{{"repository_dispatch\nevent: drift_detected"}}:::trigger
        T5{{"workflow_dispatch\nmanual trigger"}}:::trigger
    end

    subgraph Jobs ["  GitHub Actions Jobs  —  .github/workflows/mlops_pipeline.yaml  "]
        direction TB

        CI["① CI — Lint & Test\n────────────────────\nuv sync --frozen\nnbstripout check  lenient\nruff check         lenient\npytest             strict gate"]:::job

        Train["② Train Model\n────────────────────\nDVC credentials via Secrets\ndvc pull -r dagshub\npython -m rossmann_ops.train_model\nMLflow logs → DagsHub\nupload-artifact: models/"]:::job

        Simulate["③ Simulate  (manual only)\n────────────────────\nMode: schema | attack | drift\nattack: continue-on-error=true\n  exit 1 = detection confirmed"]:::job

        Build["④ Build & Push Images\n────────────────────\ndownload-artifact: models/\ndownload: data/raw/store.csv\nDockerfile.api → miles25420/rossmann-api\nDockerfile.ui  → miles25420/rossmann-ui\npush branch → :latest\npush tag    → :vX.Y.Z + :latest"]:::job

        Release["⑤ GitHub Release  (tags only)\n────────────────────\nRewrite k8s manifests\n  :latest → :vX.Y.Z\nBuild deployment-bundle.zip\n  src/  k8s/  configs/  Justfile\n  uv.lock  pyproject.toml\n  store_target_means.json\nAttach to GitHub Release"]:::job
    end

    subgraph CT ["  Continuous Training Loop  "]
        direction LR
        Drift(["simulate_production.py  --mode drift\n────────────────────\nKS-Test on CompetitionDistance\np-value < 0.05  →  drift detected"]):::ctNode
        Webhook(["GitHub API\nPOST /repos/{owner}/{repo}/dispatches\n{ event_type: drift_detected }\nRequires: GITHUB_PAT  (repo scope)"]):::ctNode
    end

    subgraph Artifacts ["  Artifact Destinations  "]
        direction LR
        DH[("DagsHub\nMLflow run logged\nExperiment tracked")]:::artifact
        DHu[("DockerHub\nmiles25420/rossmann-api\nmiles25420/rossmann-ui")]:::artifact
        GHR[("GitHub Release\ndeployment-bundle.zip\napi.yaml  ·  ui.yaml")]:::artifact
    end

    T1 -->|"every push"| CI
    T2 -->|"main / tag"| CI
    T3 -->|"tag only"| CI
    T4 -->|"drift event"| CI
    T5 -->|"manual"| CI

    CI -->|"tests pass — main / tag / dispatch"| Train
    CI -->|"tests pass — workflow_dispatch"| Simulate

    Train -->|"models/ uploaded to artifact store"| Build
    Train --> DH

    Build --> DHu
    Build -->|"tag push only"| Release
    Release --> GHR

    Drift -->|"p < 0.05"| Webhook
    Webhook -->|"fires repository_dispatch"| T4
```

---

## Getting Started

```bash
git clone https://github.com/miles2542/rossmann-ops.git
cd rossmann-ops
```

---

## Prerequisites

This project uses [`just`](https://github.com/casey/just) as its unified task runner and [`uv`](https://docs.astral.sh/uv/) for Python dependency management. Install **only these two tools** manually — `just setup` / `just deploy-all` handle everything else automatically (`.venv` creation, dependency install, DVC pull, model training, Docker builds, K8s deploy).

### Step 0 — Install `just`

`just` is a lightweight, cross-platform command runner. Install it system-wide, then **restart your shell or reopen VS Code** before continuing.

| Platform             | Command                                                                                                |
| :------------------- | :----------------------------------------------------------------------------------------------------- |
| **Windows** (winget) | `winget install --id Casey.Just`                                                                       |
| **macOS** (Homebrew) | `brew install just`                                                                                    |
| **Linux**            | `curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \| bash -s -- --to ~/.local/bin` |

### Step 1 — Install `uv`

`uv` is a fast Python package and environment manager. Install it **into your system Python** (not inside a project venv) so it is available as a global command:

```bash
pip install uv

# Or via the official installer:
# Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:         curl -LsSf https://astral.sh/uv/install.sh | sh
```

> Once `uv` is installed globally, `just setup` will automatically create the project `.venv` and install all Python dependencies. No manual `pip install` or `venv` commands needed.

### Step 2 — Install Docker Desktop and Helm

Required for Docker Compose (Option B) and K8s deployment (Option C). Not needed for local-only development (Option A).

| Tool               | Installation                                                                          |
| :----------------- | :------------------------------------------------------------------------------------ |
| **Docker Desktop** | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **Helm**           | [helm.sh/docs/intro/install](https://helm.sh/docs/intro/install/)                     |

### Step 3 — Install `kind` and `kubectl` (K8s path only)

```bash
just install-k8s-tools
```

Detects your OS and uses `winget` (Windows), `brew` (macOS), or downloads official binaries (Linux). Restart your shell after completion.

### Required Credentials

Copy `.env.example` → `.env` and populate:

```bash
# DagsHub — remote DVC storage + MLflow tracking
# Repository: https://dagshub.com/miles2542/rossmann-ops
DAGSHUB_USERNAME=miles2542
DAGSHUB_PAT=<personal-access-token>

# MLflow remote tracking server (points to DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com/miles2542/rossmann-ops.mlflow
MLFLOW_TRACKING_USERNAME=miles2542
MLFLOW_TRACKING_PASSWORD=<personal-access-token>
```

> [!NOTE]
> **GitHub Actions CI/CD Secrets:** In repository Settings → Secrets and Variables → Actions, configure:
> `DAGSHUB_USERNAME`, `DAGSHUB_PAT`, `DOCKERHUB_USERNAME` (`miles25420`), `DOCKERHUB_TOKEN`, and optionally `GITHUB_PAT` (needed for the CT retraining webhook from a local machine).

> **DockerHub credentials** are only needed as GitHub Actions secrets for the CD stage — not for local deployments.

---

## Quickstart

> [!NOTE]
> Recommend **option C** for full testing (e.g. for professor)

<details>
<summary><strong>Option A — Local Development</strong> (no Docker, no K8s — fastest path)</summary>

```bash
# 1. Install deps and pull DVC-tracked data artifacts
just setup

# 2. Train the production model (logs to MLflow, ~5-10 min)
just train-prod

# 3. Start the inference API (terminal 1) and dashboard (terminal 2)
just serve-api     # FastAPI   ->  http://localhost:8000/docs
just serve-ui      # Streamlit ->  http://localhost:8501

# 4. (Optional) Inspect experiment runs in the MLflow UI
just mlflow-ui     # ->  http://localhost:5000
```

</details>

<details>
<summary><strong>Option B — Docker Compose</strong> (no K8s required)</summary>

Pulls the published images from DockerHub and runs them with a single command. Requires **Docker Desktop only** — no `kind`, `kubectl`, or `helm` needed.

```bash
just docker-up     # pull latest images and start (detached)
just docker-down   # stop and remove containers
```

| Service      | URL                             |
| :----------- | :------------------------------ |
| Streamlit UI | `http://localhost:30000`        |
| API Docs     | `http://localhost:30100/docs`   |
| API Health   | `http://localhost:30100/health` |

> [!NOTE]
> This path uses the latest published image from DockerHub (`miles25420/rossmann-api:latest`). Prometheus/Grafana monitoring is **not** included. For the full observability stack, use Option C.

</details>

<details>
<summary><strong>Option C — Full K8s Production Deployment</strong> ✦ Recommended for graders</summary>

Trains the model locally, builds Docker images, deploys to a local KinD cluster, and provisions Prometheus + Grafana monitoring. Requires Steps 0–3 of Prerequisites.

```bash
just deploy-all
```

> Estimated runtime: 10–20 minutes on a modern machine (mainly due to Docker build and loading image into KinD).

Once running, all services are available at `localhost`:

| Service      | URL                             | Credentials               |
| :----------- | :------------------------------ | :------------------------ |
| Streamlit UI | `http://localhost:30000`        | —                         |
| API Docs     | `http://localhost:30100/docs`   | —                         |
| API Health   | `http://localhost:30100/health` | —                         |
| Grafana      | `http://localhost:30200`        | `admin` / `prom-operator` |
| Prometheus   | `http://localhost:30300`        | —                         |

</details>

### Useful Maintenance Commands

```bash
just k8s-status     # Show all pods + services across namespaces
just k8s-down       # Delete the KinD cluster
just k8s-update     # Rebuild images + rolling restart deployments
just lint           # Run Ruff linter, can add --fix to auto-fix
just format         # Auto-format code (Ruff)
just test           # Run full pytest suite with coverage
```

---

## Running Observability Demos

With the K8s stack or Docker Compose running:

```bash
just demo
```

Sends 3 sequential traffic phases — normal predictions, malformed schema requests (422s), and data-poisoned payloads (`CompetitionDistance > 100,000m`) — to create visible spikes in the Grafana dashboard at `http://localhost:30200`.

![Grafana Dashboard Demo](docs/assets/grafana_demo.png)

---

## Repository Structure

![Project Structure](docs/assets/project_structure.png)

---

## Extra Documentation

| Document                                                       | Description                                                          |
| :------------------------------------------------------------- | :------------------------------------------------------------------- |
| [ML Pipeline](docs/ML_PIPELINE.md)                             | Feature engineering, modeling strategy, metrics, SHAP explainability |
| [MLOps Architecture](docs/MLOPS_ARCHITECTURE.md)               | K8s topology, deployment strategy, CI/CD/CT flow                     |
| [Observability & Security](docs/OBSERVABILITY_AND_SECURITY.md) | Defensive layers, telemetry, drift detection, Grafana dashboard      |

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for full terms.
