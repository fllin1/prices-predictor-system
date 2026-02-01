"""
Dashboard utility functions.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

from src.ingest_data import DataIngestorFactory


def load_data_from_file(
    file_path_or_object, is_uploaded: bool = False, file_ext: str | None = None
) -> pd.DataFrame:
    """
    Load data from either a file path or uploaded file object.

    Args:
        file_path_or_object: Either a file path (str/Path) or a Streamlit uploaded file object.
        is_uploaded: Whether the file is an uploaded file object.
        file_ext: Optional file extension override. If None, will be inferred from the file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If the file type is not supported.
    """
    if file_ext is None:
        if isinstance(file_path_or_object, (str, Path)):
            file_ext = Path(file_path_or_object).suffix.lower()
        else:
            # For uploaded files, need to check name
            file_ext = os.path.splitext(file_path_or_object.name)[1].lower()

    if file_ext == ".csv":
        return pd.read_csv(file_path_or_object)

    elif file_ext == ".zip":
        if is_uploaded:
            # Save uploaded ZIP to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(file_path_or_object.getvalue())
                tmp_path = tmp_file.name
            try:
                data_ingestor = DataIngestorFactory.get_data_ingestor(".zip")
                df = data_ingestor.ingest(tmp_path)
                return df
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)  # Clean up
        else:
            data_ingestor = DataIngestorFactory.get_data_ingestor(".zip")
            return data_ingestor.ingest(str(file_path_or_object))

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
