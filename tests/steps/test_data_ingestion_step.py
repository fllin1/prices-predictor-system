import zipfile

import pandas as pd
import pytest

from steps.data_ingestion_step import data_ingestion_step


@pytest.fixture
def sample_data_zip(tmp_path):
    """Create a temporary zip file with a sample CSV."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    zip_file = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_file, "w") as zipf:
        zipf.write(csv_file, arcname="data.csv")

    return str(zip_file)


def test_data_ingestion_step(sample_data_zip):
    """Test that data ingestion reads the zip and returns a DataFrame."""
    df = data_ingestion_step(file_path=sample_data_zip)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
