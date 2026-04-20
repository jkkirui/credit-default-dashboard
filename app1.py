import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset
# -------------------------
url = "https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv"
df = pd.read_csv(url)

# -------------------------
# Prepare data
# -------------------------
X = df[['Income','Age','Loan','Loan to Income']]
y = df['Default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# -------------------------
# App UI
# -------------------------
st.title("💳 Credit Default Prediction Dashboard")

st.write("Enter customer details to predict default risk")

# Inputs
income = st.number_input("Income", min_value=0.0)
age = st.number_input("Age", min_value=18)
loan = st.number_input("Loan Amount", min_value=0.0)

loan_income = loan / income if income != 0 else 0

# Prediction button
if st.button("Predict"):
    input_data = np.array([[income, age, loan, loan_income]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default ({prob:.2f})")

# -------------------------
# Visualization
# -------------------------
st.subheader("📊 Dataset Overview")
st.write(df.head())

st.subheader("📈 Correlation Heatmap")
st.write(df.corr())

st.line_chart(df['Income'])

st.bar_chart(df['Default'].value_counts())

st.sidebar.header("User Input")