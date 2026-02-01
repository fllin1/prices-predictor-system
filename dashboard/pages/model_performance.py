"""
Model Performance page for the dashboard.

Displays MLflow experiment results, metrics comparison,
best model highlights, and training history.
"""

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import streamlit as st
from zenml.client import Client


def render():
    """Render the Model Performance page."""
    st.header("📈 Model Performance")

    try:
        # Connect to ZenML/MLflow
        _client = Client()
        _display_experiments()

    except Exception as e:
        st.error(f"Error connecting to ZenML/MLflow: {e}")


def _display_experiments():
    """Display MLflow experiments and runs."""
    try:
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]

        if not experiment_names:
            st.warning("No MLflow experiments found. Run a training pipeline first.")
            return

        selected_exp = st.selectbox("Select MLflow Experiment", experiment_names)

        if selected_exp:
            experiment = mlflow.get_experiment_by_name(selected_exp)
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            if runs.empty:
                st.info("No runs found for this experiment.")
                return

            # Find metric columns
            metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
            r2_col, rmse_col = _find_metric_columns(metric_cols)

            _display_best_model(runs, r2_col, rmse_col)
            st.divider()

            _display_training_timeline(runs, r2_col)
            st.divider()

            _display_metric_comparison(runs, metric_cols)
            st.divider()

            _display_run_details(runs, metric_cols)
            st.divider()

            _display_all_runs(runs)

    except Exception as e:
        st.warning(f"Could not fetch MLflow experiments: {e}")
        st.info("Ensure you have run a pipeline and MLflow is configured.")


def _find_metric_columns(metric_cols):
    """Find R2 and RMSE columns from metric columns."""
    r2_col = None
    rmse_col = None

    for col in metric_cols:
        if "r2" in col.lower():
            r2_col = col
        if "rmse" in col.lower():
            rmse_col = col

    return r2_col, rmse_col


def _display_best_model(runs, r2_col, rmse_col):
    """Display best model highlight section."""
    st.subheader("🏆 Best Model")

    if r2_col and not runs[r2_col].isna().all():
        best_idx = runs[r2_col].idxmax()
        best_run = runs.loc[best_idx]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best R² Score", f"{best_run[r2_col]:.4f}")
        with col2:
            if rmse_col:
                st.metric("RMSE", f"{best_run[rmse_col]:.2f}")
        with col3:
            st.metric("Run ID", best_run["run_id"][:8] + "...")
        with col4:
            if "start_time" in runs.columns:
                st.metric(
                    "Date", pd.to_datetime(best_run["start_time"]).strftime("%Y-%m-%d")
                )
    else:
        st.info("No R² metric found in runs.")


def _display_training_timeline(runs, r2_col):
    """Display training history timeline."""
    st.subheader("📅 Training History Timeline")

    if "start_time" not in runs.columns or not r2_col:
        st.info("Timeline requires start_time and R² metric columns.")
        return

    timeline_df = runs[["start_time", r2_col]].dropna()
    timeline_df = timeline_df.sort_values("start_time")
    timeline_df["start_time"] = pd.to_datetime(timeline_df["start_time"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeline_df["start_time"], timeline_df[r2_col], marker="o", linestyle="-")
    ax.set_xlabel("Date")
    ax.set_ylabel("R² Score")
    ax.set_title("Model Performance Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _display_metric_comparison(runs, metric_cols):
    """Display metric comparison across runs."""
    st.subheader("📊 Metric Comparison Across Runs")

    if not metric_cols:
        st.info("No metrics found to compare.")
        return

    # Create comparison chart
    comparison_df = runs[["run_id"] + metric_cols].copy()
    comparison_df["run_id"] = comparison_df["run_id"].str[:8]
    comparison_df = comparison_df.set_index("run_id")

    # Select metrics to compare
    default_metrics = metric_cols[:2] if len(metric_cols) >= 2 else metric_cols
    selected_metrics = st.multiselect(
        "Select metrics to compare:", metric_cols, default=default_metrics
    )

    if selected_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        comparison_df[selected_metrics].plot(kind="bar", ax=ax)
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Value")
        ax.set_title("Metric Comparison")
        ax.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _display_run_details(runs, metric_cols):
    """Display detailed information for a selected run."""
    st.subheader("🔍 Run Details")

    run_ids = runs["run_id"].tolist()

    # Build run options with timestamps if available
    if "start_time" in runs.columns:
        run_options = [
            f"{rid[:8]}... ({runs[runs['run_id'] == rid]['start_time'].values[0]})"
            for rid in run_ids
        ]
    else:
        run_options = [rid[:8] for rid in run_ids]

    selected_run_idx = st.selectbox(
        "Select a run to view details:",
        range(len(run_ids)),
        format_func=lambda x: run_options[x],
    )

    if selected_run_idx is not None:
        selected_run = runs.iloc[selected_run_idx]

        with st.expander("Parameters", expanded=True):
            param_cols = [col for col in runs.columns if col.startswith("params.")]
            if param_cols:
                params_df = pd.DataFrame(
                    {
                        "Parameter": [col.replace("params.", "") for col in param_cols],
                        "Value": [selected_run[col] for col in param_cols],
                    }
                )
                st.dataframe(params_df, hide_index=True)
            else:
                st.info("No parameters logged for this run.")

        with st.expander("Metrics", expanded=True):
            if metric_cols:
                metrics_df = pd.DataFrame(
                    {
                        "Metric": [col.replace("metrics.", "") for col in metric_cols],
                        "Value": [selected_run[col] for col in metric_cols],
                    }
                )
                st.dataframe(metrics_df, hide_index=True)
            else:
                st.info("No metrics logged for this run.")

        with st.expander("Artifacts"):
            if "artifact_uri" in runs.columns:
                st.code(selected_run["artifact_uri"])
            else:
                st.info("No artifact URI available.")


def _display_all_runs(runs):
    """Display table of all runs."""
    st.subheader("📋 All Runs")
    st.dataframe(runs, use_container_width=True)
