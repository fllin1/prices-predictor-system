# Prices Predictor System

An End-to-End MLOps pipeline for predicting house prices, built with [ZenML](https://zenml.io/) and [MLflow](https://mlflow.org/).

- [Prices Predictor System](#prices-predictor-system)
  - [📖 About the Project](#-about-the-project)
  - [🛠 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [1. Install Dependencies](#1-install-dependencies)
    - [2. Configure ZenML \& MLflow Stack](#2-configure-zenml--mlflow-stack)
  - [🧩 Pipeline Steps](#-pipeline-steps)
    - [1. Training Pipeline (`training_pipeline.py`)](#1-training-pipeline-training_pipelinepy)
    - [2. Deployment \& Inference Pipeline (`deployment_pipeline.py`)](#2-deployment--inference-pipeline-deployment_pipelinepy)
  - [🚀 How to Run](#-how-to-run)
    - [Training Pipeline](#training-pipeline)
    - [Deployment \& Inference Pipeline](#deployment--inference-pipeline)
  - [🐳 Running with Docker](#-running-with-docker)
    - [Prerequisites (Docker)](#prerequisites-docker)
    - [Using Docker Compose](#using-docker-compose)
    - [Using Docker CLI Manually](#using-docker-cli-manually)
  - [📊 Monitoring \& Dashboards](#-monitoring--dashboards)
    - [Streamlit Dashboard](#streamlit-dashboard)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![ZenML](https://img.shields.io/badge/zenml-%23460667.svg?style=for-the-badge&logo=zenml&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

## 📖 About the Project

This project implements a robust machine learning pipeline for predicting real estate prices. It demonstrates industry best practices for MLOps by automating the entire lifecycle of a machine learning model—from data ingestion to deployment.

The system utilizes **ZenML** as the infrastructure-agnostic orchestrator and **MLflow** for experiment tracking and model deployment.

Key features:

- **Automated Pipeline**: Fully automated workflow from data ingestion to model inference.
- **Experiment Tracking**: All metrics, parameters, and artifacts are logged using MLflow.
- **Continuous Deployment**: Automatically deploys the model if it meets performance criteria.
- **Reproducibility**: Versioned data, code, and models ensure every run is reproducible.

## 🛠 Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Docker (optional, but recommended for advanced ZenML stacks)

### 1. Install Dependencies

Initialize the environment and install dependencies:

```bash
uv init
uv sync
```

### 2. Configure ZenML & MLflow Stack

To run the pipeline locally with MLflow tracking and deployment, you need to register the appropriate stack components in ZenML.

```bash
# Install the MLflow integration
zenml integration install mlflow -y

# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register the MLflow model deployer
zenml model-deployer register mlflow --flavor=mlflow

# Register and set the stack
zenml stack register local-mlflow-stack \
    -a default \
    -o default \
    -d mlflow \
    -e mlflow_tracker \
    --set
```

## 🧩 Pipeline Steps

The project consists of two main pipelines:

### 1. Training Pipeline (`training_pipeline.py`)

This pipeline handles the data processing and model training lifecycle:

1. **Data Ingestion** (`data_ingestion_step.py`):
    - Reads data from the source archive.
    - Returns a pandas DataFrame.
2. **Handling Missing Values** (`handle_missing_values_step.py`):
    - Identifies and fills missing values using appropriate imputation strategies.
3. **Feature Engineering** (`feature_engineering_step.py`):
    - Transforms raw data (e.g., Log transformation) and selects features (`Gr Liv Area`, `SalePrice`).
4. **Outlier Detection** (`outlier_detection_step.py`):
    - Detects and handles outliers in the `SalePrice` column.
5. **Data Splitting** (`data_splitter_step.py`):
    - Splits the data into training and testing sets.
6. **Model Building** (`model_building_step.py`):
    - Trains the machine learning model.
7. **Model Evaluation** (`model_evaluator_step.py`):
    - Evaluates the model on the test set (MSE, RMSE, R2) and logs metrics to MLflow.

### 2. Deployment & Inference Pipeline (`deployment_pipeline.py`)

This pipeline handles continuous deployment and inference:

1. **Continuous Deployment Pipeline**:
    - Runs the **Training Pipeline** (steps above).
    - **Model Deployment**: Uses the built-in `mlflow_model_deployer_step` to deploy the trained model as a prediction service if it meets the criteria.

2. **Inference Pipeline**:
    - **Dynamic Importer** (`dynamic_importer.py`): Loads batch data for inference.
    - **Prediction Service Loader** (`prediction_service_loader.py`): Loads the active prediction service from the MLflow model deployer.
    - **Predictor** (`predictor.py`): Runs predictions on the batch data using the deployed service.

## 🚀 How to Run

### Training Pipeline

Run the training pipeline to train a new model and log experiments:

```bash
uv run run_pipeline.py
```

After running, you can inspect the experiment results in the MLflow UI:

```bash
mlflow ui --backend-store-uri "file:./mlruns"
```

*Note: The script output will provide the exact command with the correct URI.*

### Deployment & Inference Pipeline

Run the continuous deployment pipeline. This will train the model, evaluate it, and if it's better than the threshold, deploy it as a local REST API. It also runs the inference pipeline against the deployed model.

```bash
uv run run_deployment.py
```

To **stop** the prediction service:

```bash
uv run run_deployment.py --stop-service
```

## 🐳 Running with Docker

You can also run the project using Docker to ensure a reproducible environment.

### Prerequisites (Docker)

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, for simplified commands).

### Using Docker Compose

1. **Run the Training Pipeline**:

    ```bash
    docker compose up training
    ```

2. **Run the Deployment Pipeline**:

    ```bash
    docker compose up deployment
    ```

3. **Rebuild the Image**:
    If you modify dependencies or code, rebuild the image:

    ```bash
    docker compose build
    ```

### Using Docker CLI Manually

1. **Build the Image**:

    ```bash
    docker build -t prices-predictor .
    ```

2. **Run the Training Pipeline**:

    ```bash
    docker run \
      -v $(pwd)/mlruns:/app/mlruns \
      -v $(pwd)/data:/app/data \
      prices-predictor run_pipeline.py
    ```

## 📊 Monitoring & Dashboards

- **MLflow UI**: Visit the URL provided by the `mlflow ui` command (usually `http://127.0.0.1:5000`) to view experiment runs, compare metrics, and see model artifacts.
- **ZenML Dashboard**: You can verify your stack and runs using the ZenML dashboard:

    ```bash
    zenml up
    ```

### Streamlit Dashboard

Explore the data, check model performance, and use the prediction service via an interactive dashboard:

```bash
uv run streamlit run dashboard.py
```
