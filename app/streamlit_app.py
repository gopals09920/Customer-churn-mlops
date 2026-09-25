import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn & Retention Predictor")
st.write("Enter customer details below to predict churn risk.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    payload = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "SeniorCitizen": int(senior_citizen),
        "Contract_One_year": 1 if contract == "One year" else 0,
        "Contract_Two_year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
        "PaymentMethod_Credit_card_automatic": 1 if payment_method == "Credit card" else 0,
        "PaymentMethod_Electronic_check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed_check": 1 if payment_method == "Mailed check" else 0
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            res = response.json()
            st.subheader("Results:")
            
            if res["churn_prediction"] == "Yes":
                st.error(f"⚠️ Churn Risk: HIGH ({res['churn_probability']*100:.1f}%)")
            else:
                st.success(f"✅ Churn Risk: LOW ({res['churn_probability']*100:.1f}%)")
                
            st.json(res)
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")