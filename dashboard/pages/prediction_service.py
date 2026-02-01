"""
Prediction Service page for the dashboard.

Provides single and batch prediction functionality with
organized input forms and prediction history tracking.
"""

import pandas as pd
import streamlit as st
from zenml import Model

from dashboard.config import (
    EXPECTED_COLUMNS,
    FEATURE_CATEGORIES,
    SAMPLE_VALUES,
)


def render():
    """Render the Prediction Service page."""
    st.header("🔮 Prediction Service")

    # Create tabs for single and batch prediction
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        _render_single_prediction()

    with tab2:
        _render_batch_prediction()


def _render_single_prediction():
    """Render the single prediction form and results."""
    st.subheader("Single Property Prediction")
    st.write("Enter the details of the property to get a price prediction.")

    # Sample data buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(
            "📝 Load Sample Data", help="Fill form with typical property values"
        ):
            st.session_state.use_sample = True
            st.rerun()
    with col2:
        if st.button("🔄 Clear Form", help="Reset all values to zero"):
            st.session_state.use_sample = False
            st.rerun()

    use_sample = st.session_state.get("use_sample", False)

    # Create input form with categories
    with st.form("prediction_form"):
        inputs = _build_input_form(use_sample)
        submit_button = st.form_submit_button(
            "🎯 Predict Price", use_container_width=True
        )

    if submit_button:
        _handle_single_prediction(inputs)

    # Display prediction history
    _display_prediction_history()


def _build_input_form(use_sample: bool) -> dict:
    """Build the categorized input form and return inputs dictionary."""
    inputs = {}

    # Use tabs for categories
    category_tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))

    for tab, (category, features) in zip(category_tabs, FEATURE_CATEGORIES.items()):
        with tab:
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    default_val = SAMPLE_VALUES.get(feature, 0.0) if use_sample else 0.0
                    inputs[feature] = st.number_input(
                        f"{feature}", value=float(default_val), key=f"input_{feature}"
                    )

    return inputs


def _handle_single_prediction(inputs: dict):
    """Handle single prediction submission."""
    try:
        # Prepare input data
        input_df = pd.DataFrame([inputs])

        # Load Model
        model_name = "prices_predictor"
        try:
            model = Model(name=model_name, version="production")
            pipeline = model.load_artifact("sklearn_pipeline")

            # Make prediction
            prediction = pipeline.predict(input_df)
            predicted_price = prediction[0]

            st.success(f"🏷️ Estimated Price: **${predicted_price:,.2f}**")

            # Add to prediction history
            st.session_state.prediction_history.append(
                {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_price": f"${predicted_price:,.2f}",
                    "gr_liv_area": inputs.get("Gr Liv Area", 0),
                    "overall_qual": inputs.get("Overall Qual", 0),
                    "year_built": inputs.get("Year Built", 0),
                }
            )

        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")
            st.info(
                "Ensure the model 'prices_predictor' is registered and has a "
                "'production' version with 'sklearn_pipeline' artifact."
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")


def _display_prediction_history():
    """Display prediction history for the current session."""
    if not st.session_state.prediction_history:
        return

    st.divider()
    st.subheader("📜 Prediction History (This Session)")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()


def _render_batch_prediction():
    """Render the batch prediction interface."""
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV file with multiple properties to get predictions for all of them."
    )

    # Download template
    _display_template_download()

    st.divider()

    # Upload file for batch prediction
    batch_file = st.file_uploader(
        "Upload CSV file for batch prediction", type=["csv"], key="batch_uploader"
    )

    if batch_file is not None:
        _handle_batch_file(batch_file)


def _display_template_download():
    """Display template download button."""
    st.markdown("**Download Template:**")
    template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=template_csv,
        file_name="prediction_template.csv",
        mime="text/csv",
    )


def _handle_batch_file(batch_file):
    """Handle batch file upload and prediction."""
    try:
        batch_df = pd.read_csv(batch_file)
        st.success(f"Loaded {len(batch_df)} records from {batch_file.name}")

        with st.expander("Preview Data"):
            st.dataframe(batch_df.head(10))

        # Check columns
        missing_cols = set(EXPECTED_COLUMNS) - set(batch_df.columns)
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}")
            return

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            _run_batch_prediction(batch_df)

    except Exception as e:
        st.error(f"Error loading file: {e}")


def _run_batch_prediction(batch_df: pd.DataFrame):
    """Run batch prediction on the uploaded dataframe."""
    try:
        model_name = "prices_predictor"
        model = Model(name=model_name, version="production")
        pipeline = model.load_artifact("sklearn_pipeline")

        # Ensure correct column order
        input_df = batch_df[EXPECTED_COLUMNS]

        # Make predictions
        with st.spinner("Running predictions..."):
            predictions = pipeline.predict(input_df)

        # Add predictions to dataframe
        result_df = batch_df.copy()
        result_df["Predicted_Price"] = predictions
        result_df["Predicted_Price_Formatted"] = result_df["Predicted_Price"].apply(
            lambda x: f"${x:,.2f}"
        )

        st.success(f"Successfully predicted prices for {len(result_df)} properties!")

        # Display results
        _display_batch_results(result_df)

    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
        st.info(
            "Ensure the model 'prices_predictor' is registered and has a 'production' version."
        )


def _display_batch_results(result_df: pd.DataFrame):
    """Display batch prediction results with summary and download."""
    st.subheader("Results")
    st.dataframe(result_df, use_container_width=True)

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Price", f"${result_df['Predicted_Price'].mean():,.2f}")
    with col2:
        st.metric("Min Price", f"${result_df['Predicted_Price'].min():,.2f}")
    with col3:
        st.metric("Max Price", f"${result_df['Predicted_Price'].max():,.2f}")

    # Download results
    result_csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=result_csv,
        file_name="prediction_results.csv",
        mime="text/csv",
    )
