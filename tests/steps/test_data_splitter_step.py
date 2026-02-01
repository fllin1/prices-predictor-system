import pandas as pd

from steps.data_splitter_step import data_splitter_step


def test_data_splitter_step():
    """Test data splitting into train and test sets."""
    df = pd.DataFrame(
        {"feature1": range(100), "feature2": range(100), "SalePrice": range(100)}
    )

    X_train, X_test, y_train, y_test = data_splitter_step(df, target_column="SalePrice")

    # Check shapes
    assert len(X_train) + len(X_test) == 100
    assert len(y_train) + len(y_test) == 100
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Check target column is removed from features
    assert "SalePrice" not in X_train.columns
    assert "SalePrice" not in X_test.columns
