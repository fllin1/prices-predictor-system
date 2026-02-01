# Dockerization Strategy

This document outlines the strategy for containerizing the Prices Predictor System to ensure consistent execution environments and simplified deployment.

## Overview

The goal is to encapsulate the project dependencies and code into a Docker container. This ensures that the ML pipelines run in a reproducible environment, independent of the host machine's configuration.

**Note on Data Handling**: This strategy uses volume mounting for the dataset. The `data/` directory is **not** baked into the Docker image. This allows you to update data without rebuilding the image and keeps the image size smaller.

## Dockerfile Construction

We use a single-stage build leveraging `uv` for fast dependency installation.

### Base Image

- **Image**: `python:3.12-slim`
- **Reason**: Lightweight and matches the project's Python version requirement (`>=3.12`).

### Dependency Management

- **Tool**: `uv`
- **Process**:
    1. Install `uv` in the container.
    2. Copy `pyproject.toml` and `uv.lock`.
    3. Run `uv sync --frozen` to install dependencies into a virtual environment.

### Build Steps

1. **Setup**: Start from `python:3.12-slim`.
2. **System Dependencies**: Install necessary system libraries (e.g., `curl`).
3. **Project Dependencies**: Install dependencies using `uv`.
4. **ZenML Integration**: Install and configure ZenML integrations (MLflow) and register the stack within the container build process.
5. **Source Code**: Copy the project source code (`src/`, `steps/`, `pipelines/`, `run_pipeline.py`, etc.) into the container.
6. **Entrypoint**: Set the entrypoint to `python`.

## Docker Compose

We use `docker-compose` to simplify running the pipelines and managing volumes.

- **Services**:
  - `training`: Runs the training pipeline (`run_pipeline.py`).
  - `deployment`: Runs the deployment pipeline (`run_deployment.py`).
- **Volumes**:
  - `./mlruns:/app/mlruns`: Persists experiment data and model artifacts.
  - `./data:/app/data`: Mounts the local data directory into the container.

## Usage

### Using Docker Compose

1. **Build the image**:

    ```bash
    docker compose build
    ```

2. **Run the training pipeline**:

    ```bash
    docker compose up training
    ```

3. **Run the deployment pipeline**:

    ```bash
    docker compose up deployment
    ```

### Using Docker CLI Manually

If you prefer running `docker` commands directly, ensure you mount both the `mlruns` and `data` directories:

```bash
# Build the image
docker build -t prices-predictor .

# Run the training pipeline
docker run \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/data:/app/data \
    prices-predictor run_pipeline.py
```
