"""
Prices Predictor System Dashboard

Main entry point for the Streamlit dashboard application.
Run with: streamlit run dashboard.py
"""

import streamlit as st

from dashboard.pages import (
    render_data_explorer,
    render_model_performance,
    render_prediction_service,
)

# Set page configuration
st.set_page_config(
    page_title="Prices Predictor Dashboard", page_icon="🏠", layout="wide"
)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Data Explorer"

# Initialize session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Initialize session state for sample data
if "sample_data" not in st.session_state:
    st.session_state.sample_data = None

# Title
st.title("🏠 Prices Predictor System Dashboard")

# Sidebar button navigation
st.sidebar.header("Navigation")

if st.sidebar.button(
    "📊 Data Explorer",
    use_container_width=True,
    type="primary" if st.session_state.page == "Data Explorer" else "secondary",
):
    st.session_state.page = "Data Explorer"

if st.sidebar.button(
    "📈 Model Performance",
    use_container_width=True,
    type="primary" if st.session_state.page == "Model Performance" else "secondary",
):
    st.session_state.page = "Model Performance"

if st.sidebar.button(
    "🔮 Prediction Service",
    use_container_width=True,
    type="primary" if st.session_state.page == "Prediction Service" else "secondary",
):
    st.session_state.page = "Prediction Service"

st.sidebar.divider()
st.sidebar.caption(f"Current Page: **{st.session_state.page}**")

# Page routing
page = st.session_state.page

if page == "Data Explorer":
    render_data_explorer()
elif page == "Model Performance":
    render_model_performance()
elif page == "Prediction Service":
    render_prediction_service()
