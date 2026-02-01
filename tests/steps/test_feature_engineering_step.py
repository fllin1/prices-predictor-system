import pandas as pd

from steps.feature_engineering_step import feature_engineering_step


def test_feature_engineering_step():
    """Test feature engineering transformation and selection."""
    df = pd.DataFrame(
        {
            "Gr Liv Area": [100, 200, 300],
            "SalePrice": [1000, 2000, 3000],
            "OtherCol": [1, 2, 3],
        }
    )

    transformed_df = feature_engineering_step(
        df, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Check if only selected features remain
    assert list(transformed_df.columns) == ["Gr Liv Area", "SalePrice"]
    # Check if log transformation was applied (values should be smaller)
    assert (transformed_df["Gr Liv Area"] < df["Gr Liv Area"]).all()
    assert (transformed_df["SalePrice"] < df["SalePrice"]).all()
