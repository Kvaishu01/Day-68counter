import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dice_ml

# -------------------------------
# 🎯 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Day 68 - Counterfactual Explanation Generator",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Day 68: Counterfactual Explanation Generator (Explainable AI)")
st.markdown("""
This demo shows how **Counterfactual Explanations** can help us understand model predictions —  
by finding the *minimal changes* needed to alter a model’s decision.
""")

# -------------------------------
# 🧠 LOAD DATA
# -------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Sample smaller data for speed
X = X.sample(500, random_state=42)
y = y.loc[X.index]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 🌲 MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.success("✅ Random Forest model trained successfully!")

# -------------------------------
# 📊 SAMPLE PREDICTION
# -------------------------------
sample_idx = st.slider("Select a sample index", 0, len(X_test) - 1, 0)
sample = X_test.iloc[[sample_idx]]

prediction = model.predict(sample)[0]
st.write("### 🔎 Model Prediction:")
st.write(f"**Predicted Class:** {'Malignant (1)' if prediction == 1 else 'Benign (0)'}")

# -------------------------------
# 💡 COUNTERFACTUAL EXPLANATION
# -------------------------------
st.write("### 💡 Generating Counterfactual Explanation...")

# Combine training data and target
train_df = X_train.copy()
train_df['target'] = y_train

# Wrap model for DiCE
dice_data = dice_ml.Data(
    dataframe=train_df,
    continuous_features=list(X.columns),
    outcome_name='target'
)

dice_model = dice_ml.Model(model=model, backend="sklearn")

# Initialize explainer
exp = dice_ml.Dice(dice_data, dice_model)

# Prepare query instance
# Prepare query instance WITHOUT target column
query_instance = sample.copy()  # Do NOT add 'target' column

# Generate counterfactuals with spinner
with st.spinner("⏳ Generating counterfactuals..."):
    try:
        dice_exp = exp.generate_counterfactuals(
            query_instance,
            total_CFs=2,
            desired_class="opposite"
        )
        st.success("✅ Counterfactuals generated!")
        st.write("### 🧩 Counterfactual Explanation")
        st.dataframe(dice_exp.cf_examples_list[0].final_cfs_df)
    except Exception as e:
        st.error(f"⚠️ Could not generate counterfactuals: {e}")


# -------------------------------
# 📘 ABOUT
# -------------------------------
st.markdown("""
---
✅ **Key Idea:** Counterfactual explanations show *how* a prediction could change —  
for example, *"If the mean radius decreased slightly, the tumor would be classified as benign."*

🧠 **Library Used:** [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE)  
📈 **Model Used:** Random Forest Classifier  
📊 **Dataset:** Breast Cancer (Scikit-learn)
""")

