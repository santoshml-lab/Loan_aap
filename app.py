import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Loan Prediction App")

with st.form("loan_form"):

    no_of_dependents = st.number_input("No of Dependents", 0, 10)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Income Annum")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Term")
    cibil_score = st.number_input("CIBIL Score")
    residential_assets_value = st.number_input("Residential Assets Value")
    commercial_assets_value = st.number_input("Commercial Assets Value")
    luxury_assets_value = st.number_input("Luxury Assets Value")
    bank_asset_value = st.number_input("Bank Asset Value")

    submit = st.form_submit_button("Predict")

if submit:

    # 🟢 SAFE MAPPING (NO LABELENCODER)
    education_map = {"Graduate": 1, "Not Graduate": 0}
    emp_map = {"Yes": 1, "No": 0}

    input_dict = {
        "no_of_dependents": no_of_dependents,
        "education": education_map.get(education.strip(), 0),
        "self_employed": emp_map.get(self_employed.strip(), 0),
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    df = pd.DataFrame([input_dict])

    # 🟢 COLUMN ALIGNMENT (CRITICAL)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    st.success(f"Prediction: {prediction}")
