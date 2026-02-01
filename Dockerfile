# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen: Sync with lock file
# --no-dev: Do not install development dependencies
RUN uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Install the MLflow integration
RUN zenml integration install mlflow -y
# Register the MLflow experiment tracker
RUN zenml experiment-tracker register mlflow_tracker --flavor=mlflow
# Register the MLflow model deployer
RUN zenml model-deployer register mlflow --flavor=mlflow
# Register and set the stack
RUN zenml stack register local-mlflow-stack \
    -a default \
    -o default \
    -d mlflow \
    -e mlflow_tracker \
    --set

# Copy application code
COPY src/ ./src/
COPY steps/ ./steps/
COPY pipelines/ ./pipelines/
COPY docs/ ./docs/
COPY run_pipeline.py .
COPY run_deployment.py .
COPY sample_predict.py .
COPY config.yaml .

# Set the entrypoint to python
ENTRYPOINT ["python"]

# Default command runs the training pipeline
CMD ["run_pipeline.py"]
