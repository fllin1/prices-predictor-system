# Prices Predictor System

MLOps pipeline for predicting prices of houses.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## Installation

```bash
uv init
uv sync
```

### Register MLflow Stack

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Steps

1. Data Ingestion (["./steps/data_ingestion_step.py"](./steps/data_ingestion_step.py))
2. Handling Missing Values (["./steps/handle_missing_values_step.py"](./steps/handle_missing_values_step.py))
3. Feature Engineering (["./steps/feature_engineering_step.py"](./steps/feature_engineering_step.py))
4. Outlier Detection (["./steps/outlier_detection_step.py"](./steps/outlier_detection_step.py))
5. Data Splitting (["./steps/data_splitter_step.py"](./steps/data_splitter_step.py))
6. Model Building (["./steps/model_building_step.py"](./steps/model_building_step.py))
7. Model Evaluation (["./steps/model_evaluator_step.py"](./steps/model_evaluator_step.py))
8. Model Deployment (["./steps/model_deployment_step.py"](./steps/model_deployment_step.py))
