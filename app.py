import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Load Pre-trained KMeans Model and Scaler ---
model_path = r"C:\\Users\\praveen\\OneDrive\\Attachments\\Desktop\\FinalProj\\src\\kmeans_model.pkl"
scaler_path = r"C:\\Users\\praveen\\OneDrive\\Attachments\\Desktop\\FinalProj\\src\\scaler.pkl"
required_features_path = r"C:\\Users\\praveen\\OneDrive\\Attachments\\Desktop\\FinalProj\\data\\processed\\required_features.pkl"

# Load the KMeans model, scaler, and required features
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
required_features = joblib.load(required_features_path)

# --- Streamlit Interface ---
st.title("Customer Segmentation with KMeans Clustering")
st.write("Enter the customer's details below and click *Predict Cluster* to find their segment.")

# --- User Input Section ---
# Descriptions for input features
feature_descriptions = {
    "BALANCE": "Current outstanding balance on the customer's account.",
    "PURCHASES": "Total amount of purchases made by the customer.",
    "ONEOFF_PURCHASES": "Total value of one-time purchases.",
    "TENURE": "Number of months the customer has been with the company."
}

# Define important input features for user input
important_features = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "TENURE"]

# Create input fields for the user
with st.form(key='user_input_form'):
    user_data = {}
    for feature in important_features:
        user_data[feature] = st.number_input(
            f"{feature} ({feature_descriptions[feature]})",
            min_value=0.0,
            step=0.1
        )
    submit_button = st.form_submit_button(label='Predict Cluster')

# --- Ensure Required Features ---
if submit_button:
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_data])

    # Add missing features with default value 0 and ensure correct column order
    for feature in required_features:
        if feature not in user_df.columns:
            user_df[feature] = 0
    user_df = user_df[required_features]

    # --- Preprocess Input Data ---
    user_data_scaled = scaler.transform(user_df)

    # --- Predict Customer Segment ---
    cluster = model.predict(user_data_scaled)

    # --- Display Results ---
    st.subheader("Predicted Customer Segment")
    st.success(f"The predicted cluster for this customer is: *Cluster {cluster[0]}*")

    # --- Cluster Descriptions ---
    st.subheader("Cluster Descriptions")
    st.info(""" 
    - *Cluster 0*: High-value customers with large purchases and balanced spending.
    - *Cluster 1*: Moderate spenders with steady activity.
    - *Cluster 2*: Customers with low purchases and short tenure.
    - *Cluster 3*: Customers primarily making one-off large purchases.
    """)

    # --- Recommendations ---
    recommendations = {
        0: "Offer exclusive loyalty rewards to retain high-value customers.",
        1: "Provide promotions to encourage steady activity.",
        2: "Focus on engagement strategies to increase purchases.",
        3: "Promote frequent-purchase discounts or subscriptions."
    }
    st.subheader("Recommended Actions")
    st.info(recommendations[cluster[0]])

   

   

   

   

# --- Help Section ---
with st.expander("What is Customer Segmentation?"):
    st.write("""
    Customer segmentation is the process of dividing customers into groups based on common characteristics.
    - It helps businesses create tailored marketing strategies.
    - KMeans clustering uses customer data to automatically group customers into segments.
    """)
