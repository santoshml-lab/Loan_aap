
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Loan Prediction System")

with st.form("loan_form"):

    no_of_dependents = st.number_input("Dependents", 0, 10)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Income")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Term")
    cibil_score = st.number_input("CIBIL Score")
    residential_assets_value = st.number_input("Residential Assets")
    commercial_assets_value = st.number_input("Commercial Assets")
    luxury_assets_value = st.number_input("Luxury Assets")
    bank_asset_value = st.number_input("Bank Asset")

    submit = st.form_submit_button("Predict")

if submit:

    # EXACT SAME MAPPING AS TRAINING
    education = {"Graduate": 1, "Not Graduate": 0}[education]
    self_employed = {"Yes": 1, "No": 0}[self_employed]

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

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=columns, fill_value=0)

    pred = model.predict(df)[0]

    st.success(f"Prediction: {pred}")
    st.subheader("Feature Importance")

import matplotlib.pyplot as plt
import pandas as pd

# check feature importance exists
if hasattr(model, "feature_importances_"):

    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": columns,
        "Importance": importances
    })

    feat_df = feat_df.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])

    st.pyplot(fig)

else:
    st.warning("This model does not support feature importance")


st.subheader("📌 Reason")

# Top 3 important features
top_features = feat_df.sort_values(by="Importance", ascending=False).head(3)

reasons = []

for feature in top_features["Feature"]:
    
    if feature == "cibil_score":
        if input_dict["cibil_score"] > 700:
            reasons.append("Good CIBIL score")
        else:
            reasons.append("Low CIBIL score")

    elif feature == "income_annum":
        if input_dict["income_annum"] > 500000:
            reasons.append("High income")
        else:
            reasons.append("Low income")

    elif feature == "loan_amount":
        if input_dict["loan_amount"] > 2000000:
            reasons.append("High loan amount")
        else:
            reasons.append("Moderate loan amount")

    elif feature == "bank_asset_value":
        if input_dict["bank_asset_value"] > 500000:
            reasons.append("Strong bank assets")
        else:
            reasons.append("Low bank assets")

# show reasons
for r in reasons:
    st.write("👉", r)

