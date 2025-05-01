import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier

# Load the trained model and encoders
best_xgb = XGBClassifier()
best_xgb.load_model("best_xgb_models.json")
one_hot_columns = joblib.load("one_hot_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App Layout
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #333;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        }
        .main-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊US Bank Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their details.")

# Input Fields
with st.form("churn_form"):
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=10000, value=200, step=10)
        age = st.number_input("Age", min_value=18, max_value=100, value=70, step=1)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=40, value=1, step=1)
        balance = st.number_input("Balance (USD)", min_value=0.0, max_value=1000000.0, value=2000.0, step=100.0)
        estimated_salary = st.number_input("Estimated Salary (USD)", min_value=0.0, max_value=500000.0, value=10000.0, step=1000.0)

    with col2:
        geography = st.selectbox("Geography", ["Germany", "France", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
        has_cr_card = st.number_input("Has Credit Card (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0, step=1)
        is_active_member = st.number_input("Active Member (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0, step=1)

    submit = st.form_submit_button("Predict Churn")

# Process input when form is submitted
if submit:
    # Encode categorical variables
    gender_encoded = label_encoder.transform([gender])[0]

    # Create a dataframe for user input
    input_data = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products, 
                                has_cr_card, is_active_member, estimated_salary, 
                                gender_encoded, 
                                int(geography == "France"), 
                                int(geography == "Germany"), 
                                int(geography == "Spain")]], 
                                columns=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                                         "HasCrCard", "IsActiveMember", "EstimatedSalary", 
                                         "Gender", "Geography_France", "Geography_Germany", "Geography_Spain"])
    
    # Align features with the model
    input_data = input_data.reindex(columns=one_hot_columns, fill_value=0)
    
    # Make prediction
    prediction = best_xgb.predict(input_data)
    prediction_text = "This customer is **likely to churn.**" if prediction[0] == 1 else "This customer is **likely to stay.**"

    # Display result with UI styling
    st.success(f"### Prediction: {prediction_text}")
