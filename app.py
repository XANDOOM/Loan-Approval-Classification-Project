import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and preprocessing tools
# Ensure these 3 files are in your GitHub root folder
model = joblib.load('compressed_model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("💰 Loan Approval Prediction Tool")
st.markdown("Enter applicant details below to get an instant prediction.")

# 2. Input UI - Grouped for better UX
with st.form("loan_form"):
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (person_age)", min_value=18, max_value=100, value=25)
        income = st.number_input("Annual Income (person_income)", min_value=0, value=50000)
        gender = st.selectbox("Gender (person_gender)", ["male", "female"])
    with col2:
        education = st.selectbox("Education Level (person_education)", ["Bachelor", "Master", "High School", "Associate", "Doctorate"])
        emp_exp = st.number_input("Employment Experience in Years (person_emp_exp)", min_value=0, max_value=60, value=5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    st.subheader("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_amnt = st.number_input("Loan Amount (loan_amnt)", min_value=500, value=10000)
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    with col4:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=11.0, step=0.1)
        prev_default = st.selectbox("Previous Default on File?", ["No", "Yes"])
    
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    cred_hist_len = st.number_input("Credit History Length (Years)", min_value=0, value=5)

    submit = st.form_submit_button("Predict Loan Status")

if submit:
    # A. Build the input DataFrame with exact original column names [cite: 144, 155, 171]
    input_data = pd.DataFrame({
        'person_age': [float(age)],
        'person_gender': [gender],
        'person_education': [education],
        'person_income': [float(income)],
        'person_emp_exp': [int(emp_exp)],
        'person_home_ownership': [home_ownership],
        'loan_amnt': [float(loan_amnt)],
        'loan_intent': [loan_intent],
        'loan_int_rate': [float(loan_int_rate)],
        'loan_percent_income': [float(loan_amnt / income)],
        'cb_person_cred_hist_length': [float(cred_hist_len)],
        'credit_score': [int(credit_score)],
        'previous_loan_defaults_on_file': [prev_default]
    })

    # B. Apply One-Hot Encoding [cite: 295]
    cat_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    encoded_features = encoder.transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))

    # C. Apply Scaling [cite: 366]
    num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    scaled_features = scaler.transform(input_data[num_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=num_cols)

    # D. Concatenate into the final feature set (27 columns) [cite: 299, 409]
    final_features = pd.concat([scaled_df, encoded_df], axis=1)

    # E. Prediction [cite: 578]
    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)[0][1]

    # F. Display Results
    if prediction[0] == 1:
        st.success(f"✅ **Approved!** (Probability of Approval: {probability:.2%})")
    else:
        st.error(f"❌ **Rejected.** (Probability of Approval: {probability:.2%})")
