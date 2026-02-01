"""
Data Explorer page for the dashboard.

Allows users to upload or select data files, view statistics,
and visualize data distributions and correlations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from dashboard.utils import load_data_from_file
from pipelines.config import DATA_DIR


def render():
    """Render the Data Explorer page."""
    st.header("📊 Data Explorer")

    st.subheader("Data Source Selection")
    data_source = st.radio(
        "Choose data source:",
        ["Upload CSV/ZIP file", "Select from local data directory"],
        horizontal=True,
    )

    df = None

    if data_source == "Upload CSV/ZIP file":
        df = _handle_file_upload()
    else:
        df = _handle_local_file_selection()

    if df is not None:
        # Store sample data for prediction page
        st.session_state.sample_data = df
        _display_dataset_overview(df)
        _display_visualizations(df)


def _handle_file_upload():
    """Handle file upload and return DataFrame."""
    uploaded_file = st.file_uploader(
        "Choose a CSV or ZIP file",
        type=["csv", "zip"],
        help="Upload a CSV file or a ZIP file containing CSV",
    )

    if uploaded_file is not None:
        try:
            df = load_data_from_file(uploaded_file, is_uploaded=True)
            st.success(f"Successfully loaded {uploaded_file.name}")
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None
    return None


def _handle_local_file_selection():
    """Handle local file selection and return DataFrame."""
    data_dir = Path(DATA_DIR)

    if not data_dir.exists():
        st.error(f"Data directory {DATA_DIR} does not exist")
        return None

    csv_files = [f.name for f in data_dir.glob("*.csv")]
    zip_files = [f.name for f in data_dir.glob("*.zip")]
    available_files = sorted(csv_files + zip_files)

    if not available_files:
        st.warning(f"No CSV or ZIP files found in {DATA_DIR}")
        return None

    selected_file = st.selectbox(
        "Select a file from data directory:",
        [""] + available_files,
        format_func=lambda x: "Choose a file..." if x == "" else x,
    )

    if selected_file and selected_file != "":
        file_path = data_dir / selected_file
        try:
            df = load_data_from_file(file_path, is_uploaded=False)
            st.success(f"Successfully loaded {selected_file}")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    return None


def _display_dataset_overview(df):
    """Display dataset overview statistics."""
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")

    with st.expander("Show Data Head"):
        st.dataframe(df.head())

    with st.expander("Show Data Statistics"):
        st.write(df.describe())


def _display_visualizations(df):
    """Display data visualizations."""
    st.subheader("Visualizations")

    # Select numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for visualization.")
        return

    col1, col2 = st.columns(2)

    with col1:
        _display_univariate_analysis(df, numeric_cols)

    with col2:
        _display_bivariate_analysis(df, numeric_cols)

    _display_correlation_matrix(df, numeric_cols)


def _display_univariate_analysis(df, numeric_cols):
    """Display univariate analysis (histogram)."""
    st.markdown("### Univariate Analysis")
    selected_col = st.selectbox("Select Column for Histogram", numeric_cols)

    if selected_col:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)


def _display_bivariate_analysis(df, numeric_cols):
    """Display bivariate analysis (scatter plot vs target)."""
    st.markdown("### Bivariate Analysis (vs Price)")

    # Determine target column
    target_col = (
        "SalePrice"
        if "SalePrice" in df.columns
        else ("Price" if "Price" in df.columns else numeric_cols[-1])
    )

    x_col = st.selectbox("Select Feature (X axis)", numeric_cols, index=0)

    if x_col:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[target_col], ax=ax)
        st.pyplot(fig)
        plt.close(fig)


def _display_correlation_matrix(df, numeric_cols):
    """Display correlation heatmap."""
    st.markdown("### Correlation Matrix")

    if st.checkbox("Show Correlation Heatmap"):
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
