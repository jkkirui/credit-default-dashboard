import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load Data
# -------------------------
url = "https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv"
df = pd.read_csv(url)

# -------------------------
# Prepare Model
# -------------------------
X = df[['Income','Age','Loan','Loan to Income']]
y = df['Default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# -------------------------
# Title
# -------------------------
st.title("🏦 Credit Risk Executive Dashboard")

# =========================
# 1. KPI PANEL
# =========================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    default_rate = df['Default'].mean() * 100
    st.metric("Default Rate", f"{default_rate:.1f}%")

with col3:
    st.metric("Avg Income", f"{df['Income'].mean():,.0f}")

with col4:
    st.metric("Avg Loan", f"{df['Loan'].mean():,.0f}")

# =========================
# 2. RISK TREND
# =========================
st.subheader("📈 Default Risk by Age Group")

# Create age groups
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[18,25,35,45,60,100],
    labels=["18-25","26-35","36-45","46-60","60+"]
)

age_risk = df.groupby('AgeGroup')['Default'].mean() * 100

st.line_chart(age_risk)


# =========================
# 3. PORTFOLIO OVERVIEW
# =========================
st.subheader("📊 Portfolio Risk Distribution")

default_counts = df['Default'].value_counts()

# Rename for clarity
default_counts.index = ['Non-Default', 'Default']

# Colors: Non-Default = blue, Default = red
colors = ['blue', 'red']

fig, ax = plt.subplots()
ax.bar(default_counts.index, default_counts.values, color=colors)

ax.set_ylabel("Number of Customers")
ax.set_title("Portfolio Risk Distribution")

st.pyplot(fig)

# =========================
# 4. PREDICTION TOOL
# =========================
st.subheader("🔍 Customer Risk Check")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Income", min_value=0.0)

with col2:
    age = st.number_input("Age", min_value=18)

with col3:
    loan = st.number_input("Loan Amount", min_value=0.0)

loan_income = loan / income if income > 0 else 0

if st.button("Check Risk"):
    input_data = pd.DataFrame([[income, age, loan, loan_income]],
                             columns=['Income','Age','Loan','Loan to Income'])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk ({prob:.2%})")