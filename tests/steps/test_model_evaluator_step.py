import mlflow
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from steps.model_evaluator_step import model_evaluator_step


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Mock MLflow to avoid actual logging."""
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: None)


def test_model_evaluator_step():
    """Test model evaluation metrics."""
    # Create a dummy model
    model = LinearRegression()
    X_test = pd.DataFrame({"feature1": [1, 2, 3]})
    y_test = pd.Series([2, 4, 6])

    # Train it perfectly so we know the metrics
    model.fit(X_test, y_test)

    r2, mse = model_evaluator_step(model, X_test, y_test)

    assert isinstance(r2, float)
    assert isinstance(mse, float)
    # Perfect fit should have R2 = 1.0 and MSE = 0.0
    assert r2 == 1.0
    assert mse == 0.0
