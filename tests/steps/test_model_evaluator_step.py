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

    # Directly call the entrypoint function of the step
    # IMPORTANT: The step implementation expects a 'Pipeline' object with a 'preprocessor' step.
    # We need to mock that structure since we are passing a raw LinearRegression model here.
    
    # Create a mock pipeline object
    class MockPipeline:
        def __init__(self, model):
            self.model = model
            self.named_steps = {
                "preprocessor": self, # Mock preprocessor to return itself
                "model": model # Mock the model step
            }
        
        def transform(self, X):
            # Mock transform to return X as is (or as numpy array if needed)
            return X
            
        def predict(self, X):
            return self.model.predict(X)

    mock_pipeline = MockPipeline(model)

    # ZenML steps return a Tuple of outputs if multiple outputs are defined.
    # The return type of `model_evaluator_step` is Tuple[dict, float] based on its signature.
    # (Though in the step definition it returns `evaluation_metrics`, `mse`)
    
    outputs = model_evaluator_step.entrypoint(
        trained_model=mock_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    
    # Check if outputs is a tuple
    if isinstance(outputs, tuple):
        metrics_dict = outputs[0]
        mse_val = outputs[1]
    else:
        # If for some reason it returns just one value (unlikely given signature)
        metrics_dict = outputs
        mse_val = None

    assert isinstance(metrics_dict, dict)
    assert isinstance(mse_val, float)
    
    # Check values
    assert metrics_dict['R-Squared'] == 1.0
    assert mse_val < 1e-10 # Approximately 0 (allowing for floating point errors)
