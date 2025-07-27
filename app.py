
import streamlit as st
import pandas as pd
import joblib

st.title("California Housing Price Prediction")

# Load the model and pipeline
model = joblib.load("best_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:", input_data)
    input_prepared = pipeline.transform(input_data)
    predictions = model.predict(input_prepared)
    st.write("Predicted Median House Values:", predictions)
