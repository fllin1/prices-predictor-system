1.  **Dashboard Update (`dashboard.py`)**:
    *   Add imports: `tempfile`, `pathlib.Path`.
    *   Create a `load_data_from_file` helper function to handle both uploaded files (objects) and local paths (strings).
        *   Handle CSV files directly with `pd.read_csv`.
        *   Handle ZIP files by saving uploads to a temp file before passing to `DataIngestorFactory`, or passing the path directly for local files.
    *   In the "Data Explorer" section:
        *   Add a `st.radio` to toggle between "Upload File" and "Select Local File".
        *   **Option 1 (Upload)**: Use `st.file_uploader`.
        *   **Option 2 (Select)**: Use `Path(DATA_DIR).glob` to list `.csv` and `.zip` files, then `st.selectbox`.
        *   Call `load_data_from_file` with the appropriate input.
    *   Ensure the dataframe `df` is correctly populated for the existing visualization logic.