import pandas as pd

from steps.outlier_detection_step import outlier_detection_step


def test_outlier_detection_step():
    """Test outlier detection removes rows."""
    # Create data with a clear outlier
    # Use enough data points so Z-score detection works reliably
    df = pd.DataFrame(
        {
            "SalePrice": [100] * 50 + [10000]  # 10000 is an outlier
        }
    )

    clean_df = outlier_detection_step(df, column_name="SalePrice")

    # Should remove the outlier
    assert len(clean_df) < len(df)
    assert 10000 not in clean_df["SalePrice"].values
