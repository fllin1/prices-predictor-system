import mlflow
import pandas as pd
import pytest

from steps.model_building_step import model_building_step


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Mock MLflow to avoid actual logging during tests."""
    monkeypatch.setattr(mlflow, "autolog", lambda: None)


def test_model_building_step():
    """Test that model building returns a trained model."""
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([10, 20, 30])

    model = model_building_step(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")
    # Check if it's a sklearn estimator (or your specific model type)
    assert hasattr(model, "fit")
