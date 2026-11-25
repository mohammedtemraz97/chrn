"""
Day 5 (Merged) - Example 2: Trainer Live Web Application (Streamlit + ML + GenAI)
File: day5_ex2_trainer_app.py

Features:

2. Real-time customer churn risk prediction (@st.cache_resource).
3. Feature importance chart explaining top churn drivers.
4. Automated retention advisor powered by Gemini 2.5 Flash (with offline fallback).

Run:
    python day5_ex1_train_and_save.py
    streamlit run day5_ex2_trainer_app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Intelligence", page_icon="📡", layout="wide")

st.title("📡 Customer Churn Prediction & Retention Advisor")
st.markdown("Cloud-ready ML application combining **serialized models**, **feature explainability**, and **AI retention strategies**.")


@st.cache_data
def load_data():
    return pd.read_csv("customer_churn.csv")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    columns = joblib.load("models/churn_columns.pkl")
    importance = joblib.load("models/churn_importance.pkl") if os.path.exists("models/churn_importance.pkl") else None
    return model, columns, importance


def generate_retention_advice(city, contract, tenure, bill, calls, prob):
    """Generate executive retention actions using Gemini 2.5 Flash or local rule engine."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client()
            prompt = (
                f"You are a telecom retention strategist. Customer details: city={city}, "
                f"contract={contract}, tenure={tenure} months, bill=${bill}, support_calls={calls}, "
                f"predicted_churn_probability={prob:.1%}. Write 3 short actionable retention steps."
            )
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return resp.text
        except Exception:
            pass

    actions = []
    if calls >= 3:
        actions.append("- **Priority Technical Support:** Assign a senior technician within 24 hours to resolve repeated complaints.")
    if contract == "Monthly":
        actions.append("- **Annual Contract Incentive:** Offer a 15% discount for 3 months upon upgrading to an Annual plan.")
    if tenure < 6:
        actions.append("- **New Customer Check-In:** Schedule a customer success follow-up call during the first 6 months.")
    if not actions:
        actions.append("- **Loyalty Bonus:** Provide a complimentary 10GB data bundle to reinforce retention.")
    return "\n".join(actions)


if not os.path.exists("customer_churn.csv") or not os.path.exists("models/churn_model.pkl"):
    st.error("Run `python day5_ex1_train_and_save.py` first to generate `models/churn_model.pkl`.")
    st.stop()

df = load_data()
model, model_columns, feature_importance = load_artifacts()

# Sidebar filters
st.sidebar.header("🔍 Filter Customer Segments")
selected_cities = st.sidebar.multiselect("City:", df["city"].unique().tolist(), default=df["city"].unique().tolist())
selected_contracts = st.sidebar.multiselect("Contract Type:", df["contract_type"].unique().tolist(), default=df["contract_type"].unique().tolist())
filtered_df = df[df["city"].isin(selected_cities) & df["contract_type"].isin(selected_contracts)]

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(filtered_df))
c2.metric("Historical Churn Rate", f"{filtered_df['churn'].mean() * 100:.1f}%" if len(filtered_df) else "0.0%")
c3.metric("Avg Monthly Bill", f"${filtered_df['monthly_bill_usd'].mean():.2f}" if len(filtered_df) else "$0.00")
c4.metric("High-Complaint Accounts (3+)", len(filtered_df[filtered_df["support_calls"] >= 3]))

st.markdown("---")
st.subheader("🔮 Real-Time Customer Churn Risk Predictor")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", ["Gaza", "Ramallah", "Nablus", "Hebron"])
        contract = st.selectbox("Contract Type", ["Monthly", "Annual", "Two-Year"])
    with col2:
        tenure = st.slider("Tenure (Months)", 1, 48, 6)
        bill = st.number_input("Monthly Bill ($)", min_value=10.0, max_value=200.0, value=65.0)
    with col3:
        calls = st.slider("Support Calls Count", 0, 5, 3)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    input_df = pd.DataFrame([{
        "city": city,
        "contract_type": contract,
        "tenure_months": tenure,
        "monthly_bill_usd": bill,
        "support_calls": calls,
    }])
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    r1, r2 = st.columns(2)
    with r1:
        if pred == 1:
            st.error(f"⚠️ High Churn Risk (Probability: {prob:.1%})")
        else:
            st.success(f"✅ Low Churn Risk (Probability: {prob:.1%})")
    with r2:
        st.info("💡 **Recommended Retention Plan:**\n\n" + generate_retention_advice(city, contract, tenure, bill, calls, prob))

st.markdown("---")
ch1, ch2, ch3 = st.columns(3)
with ch1:
    st.subheader("Churn Rate (%) by Contract")
    st.bar_chart(filtered_df.groupby("contract_type")["churn"].mean() * 100)
with ch2:
    st.subheader("Churn Rate (%) by Support Calls")
    st.bar_chart(filtered_df.groupby("support_calls")["churn"].mean() * 100)
with ch3:
    st.subheader("Top Model Drivers (Importance)")
    if feature_importance is not None:
        st.bar_chart(feature_importance.head(5))
