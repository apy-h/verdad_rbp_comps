import streamlit as st
import numpy as np
import pandas as pd
from helpers import prep_greg_data

st.title("Greg Data Viewer")
st.write("This app displays the prepared Greg data using the prep_greg_data function.")

# Add sidebar controls for optional parameters
st.sidebar.header("Data Filters")
start_date = st.sidebar.text_input("Start Date (YYYY-MM)", placeholder="e.g., 2023-01")
end_date = st.sidebar.text_input("End Date (YYYY-MM)", placeholder="e.g., 2024-12")
force_reload = st.sidebar.checkbox("Force Reload", help="Force reprocessing even if cached file exists")

# Convert empty strings to None
start_date = start_date if start_date.strip() else None
end_date = end_date if end_date.strip() else None

# Load and display data
if st.button("Load Data") or st.sidebar.button("Apply Filters"):
    with st.spinner("Loading data..."):
        try:
            df = prep_greg_data(start_date=start_date, end_date=end_date, force_reload=force_reload)

            st.success(f"Data loaded successfully! Shape: {df.shape}")

            # Display basic info
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

            # Display column info
            st.subheader("Column Information")
            st.write(f"Columns: {', '.join(df.columns.tolist())}")

            # Display the dataframe
            st.subheader("Data Preview")
            st.dataframe(df, use_container_width=True)

            # Display basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Auto-load data on first run
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True
    with st.spinner("Loading initial data..."):
        try:
            df = prep_greg_data()
            st.success(f"Initial data loaded! Shape: {df.shape}")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading initial data: {str(e)}")