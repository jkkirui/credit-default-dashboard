import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# -------------------------
# App UI
# -------------------------
st.title("💳 Credit Default Prediction Dashboard")
st.write("Enter customer details to predict default risk")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("User Input")

income = st.sidebar.number_input("Income", min_value=0.0)
age = st.sidebar.number_input("Age", min_value=18)
loan = st.sidebar.number_input("Loan Amount", min_value=0.0)

loan_income = loan / income if income > 0 else 0

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[income, age, loan, loan_income]],
                             columns=['Income','Age','Loan','Loan to Income'])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default ({prob:.2f})")

# -------------------------
# Dataset Overview
# -------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())
# -------------------------
# Income Trend
# -------------------------
st.subheader("📉 Income Trend")
st.line_chart(df['Income'])
# -------------------------
# Combined Chart: Default Rate + Age Distribution
# -------------------------
st.subheader("📊 Default Rate & Age Distribution")

fig, ax1 = plt.subplots()

# -------------------------
# Combined Chart: Default Rate + Age Distribution
# -------------------------
st.subheader("📊 Default Rate & Age Distribution")

fig, ax1 = plt.subplots()

# Histogram (Age distribution)
ax1.hist(df['Age'], bins=15, alpha=0.6)
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of People")

# Secondary axis for default rate
ax2 = ax1.twinx()
# Default rate by age
age_default_rate = df.groupby('Age')['Default'].mean() * 100
# Optional smoothing
age_default_rate = age_default_rate.rolling(window=3, min_periods=1).mean()
# Plot line (RED)
ax2.plot(age_default_rate.index, age_default_rate.values, marker='o', color='red')
ax2.set_ylabel("Default Rate (%)")
# Title
ax1.set_title("Default Risk vs Age Distribution")
st.pyplot(fig)

# -------------------------
# Default Distribution (Custom Colors)
# -------------------------
st.subheader("📊 Default Distribution")

counts = df['Default'].value_counts()

# Color: Default=1 → yellow, others → blue
colors = ['yellow' if idx == 1 else 'blue' for idx in counts.index]

fig2, ax2 = plt.subplots()
ax2.bar(counts.index.astype(str), counts.values, color=colors)

ax2.set_xlabel("Default")
ax2.set_ylabel("Count")
ax2.set_title("Default vs Non-Default")

st.pyplot(fig2)

# -------------------------
# Feature Importance
# -------------------------
st.subheader("🔍 Feature Importance")

importance = pd.Series(model.feature_importances_, index=X.columns)
st.bar_chart(importance)
# Correlation Heatmap
# -------------------------
st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
