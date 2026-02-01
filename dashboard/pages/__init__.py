"""
Dashboard pages package.
"""

from dashboard.pages.data_explorer import render as render_data_explorer
from dashboard.pages.model_performance import render as render_model_performance
from dashboard.pages.prediction_service import render as render_prediction_service

__all__ = [
    "render_data_explorer",
    "render_model_performance",
    "render_prediction_service",
]
