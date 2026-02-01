"""
Prices Predictor Dashboard Package.

A Streamlit-based dashboard for data exploration, model performance tracking,
and price predictions.
"""

from dashboard.config import (
    EXPECTED_COLUMNS,
    FEATURE_CATEGORIES,
    SAMPLE_VALUES,
)
from dashboard.utils import load_data_from_file

__all__ = [
    "EXPECTED_COLUMNS",
    "FEATURE_CATEGORIES",
    "SAMPLE_VALUES",
    "load_data_from_file",
]
