
import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
le_edu = joblib.load("le_edu.pkl")
le_emp = joblib.load("le_emp.pkl")

st.title("Loan Prediction Dashboard")

st.write("Enter details below:")

# FORM UI
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

# PREDICTION
if submit:

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

    # ENCODING (IMPORTANT)
    input_dict["education"] = le_edu.transform([input_dict["education"]])[0]
    input_dict["self_employed"] = le_emp.transform([input_dict["self_employed"]])[0]

    # DATAFRAME
    df = pd.DataFrame([input_dict])

    # FORCE SAME COLUMN ORDER AS TRAINING
    df = df.reindex(columns=columns, fill_value=0)

    # PREDICT
    prediction = model.predict(df)[0]

    st.success(f"Prediction: {prediction}")
