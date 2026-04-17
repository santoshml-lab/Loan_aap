
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
le_edu = joblib.load("le_edu.pkl")
le_emp = joblib.load("le_emp.pkl")

st.title("Loan Approval Dashboard")

no_of_dependents = st.number_input("Dependents")
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil_score = st.number_input("CIBIL Score")
residential_assets_value = st.number_input("Residential Assets")
commercial_assets_value = st.number_input("Commercial Assets")
luxury_assets_value = st.number_input("Luxury Assets")
bank_asset_value = st.number_input("Bank Assets")

if st.button("Predict"):

    input_dict = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    input_dict["education"] = le_edu.transform([input_dict["education"]])[0]
    input_dict["self_employed"] = le_emp.transform([input_dict["self_employed"]])[0]

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=columns, fill_value=0)

    pred = model.predict(df)[0]

    if pred == "Approved":
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
