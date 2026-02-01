# Dashboard Design Document

## 1. Purpose

The purpose of this Dashboard is to provide an interactive graphical user interface (GUI) for the Prices Predictor System. It serves three main objectives:

1. **Data Transparency**: Allow stakeholders to explore and understand the dataset used for training.
2. **Model Transparency**: Visualize model performance metrics and training history.
3. **Usability**: Enable non-technical users to make price predictions without using CLI scripts.

## 2. Technology Stack

We will use **Streamlit** as the core framework for the dashboard.

- **Frontend/Backend**: [Streamlit](https://streamlit.io/) (Python-based web app framework).
- **Data Processing**: `pandas`, `numpy`.
- **Visualization**: `matplotlib`, `seaborn`.
- **Model Integration**: `mlflow` (for metrics), `zenml` (for artifact retrieval), and `scikit-learn` (for inference).

Streamlit is chosen for its rapid development cycle and seamless integration with the existing Python data science stack.

## 3. How to Run

**Launch the Dashboard**:

```bash
streamlit run dashboard.py
```

## 4. Features

### A. Data Explorer Page

This page will visualize the training data statistics and distributions.

- **Dataset Overview**: Display head of the dataframe, shape, and column data types.
- **Univariate Analysis**: Histograms and boxplots for numerical features (e.g., price, weight).
- **Bivariate Analysis**: Scatter plots showing relationships between features and the target (Price).
- **Correlation Heatmap**: Visualizing feature correlations.

### B. Model Performance Page

This page will connect to MLflow to display training results.

- **Experiment Tracking**: Table showing recent runs with their parameters and metrics.
- **Metric Visualization**: Charts comparing R2 score and RMSE across different models/runs.

### C. Prediction Interface

A form to input feature values and get a real-time price prediction.

- **Input Form**: Fields for all required model features (e.g., `payment_value`, `product_weight_g`, etc.).
- **Predict Button**: Triggers the inference logic using the latest trained model.
- **Result Display**: Shows the predicted price.
