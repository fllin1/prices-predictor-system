import pandas as pd

from steps.handle_missing_values_step import handle_missing_values_step


def test_handle_missing_values_step():
    """Test missing value imputation."""
    df = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6], "C": [7, 8, 9]})

    clean_df = handle_missing_values_step(df)

    assert not clean_df.isnull().values.any()
    assert clean_df.shape == df.shape
    # Check if values were filled (simple check, specific logic depends on implementation)
    assert len(clean_df) == 3
