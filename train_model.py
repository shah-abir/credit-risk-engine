"""
Credit Risk Model Training Pipeline
- Generates realistic synthetic credit data (based on Lending Club feature distributions)
- Trains XGBoost classifier
- Generates SHAP explainer
- Saves model artifacts for the Streamlit app
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import joblib
import json
import os

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. Generate Realistic Synthetic Credit Data
# ──────────────────────────────────────────────
print("📊 Generating synthetic credit data...")

n_samples = 10000

data = {
    "loan_amount": np.random.lognormal(mean=9.5, sigma=0.7, size=n_samples).clip(1000, 40000).astype(int),
    "annual_income": np.random.lognormal(mean=11.0, sigma=0.6, size=n_samples).clip(15000, 300000).astype(int),
    "interest_rate": np.round(np.random.uniform(5.0, 28.0, n_samples), 2),
    "dti_ratio": np.round(np.random.uniform(0, 45, n_samples), 2),
    "credit_score": np.random.normal(700, 60, n_samples).clip(300, 850).astype(int),
    "employment_length": np.random.choice(range(0, 11), n_samples, p=[0.08, 0.07, 0.09, 0.08, 0.07, 0.08, 0.07, 0.09, 0.1, 0.12, 0.15]),
    "num_open_accounts": np.random.poisson(lam=10, size=n_samples).clip(1, 40),
    "num_derogatory_records": np.random.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3], n_samples),
    "revolving_utilization": np.round(np.random.beta(2, 5, n_samples) * 100, 1),
    "total_credit_lines": np.random.poisson(lam=22, size=n_samples).clip(2, 60),
    "months_since_last_delinquency": np.random.choice(
        [0, 6, 12, 18, 24, 36, 48, 60, 72, 999],
        n_samples,
        p=[0.35, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.08, 0.06, 0.15]
    ),
    "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n_samples, p=[0.4, 0.42, 0.15, 0.03]),
    "loan_purpose": np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "car", "education", "other"],
        n_samples,
        p=[0.35, 0.20, 0.10, 0.08, 0.05, 0.07, 0.05, 0.03, 0.07]
    ),
    "term_months": np.random.choice([36, 60], n_samples, p=[0.65, 0.35]),
}

df = pd.DataFrame(data)

# Generate realistic default probability based on features
default_logit = (
    -4.0
    + 0.08 * (df["interest_rate"] - 12)
    + 0.04 * (df["dti_ratio"] - 18)
    - 0.015 * (df["credit_score"] - 700)
    + 0.3 * df["num_derogatory_records"]
    + 0.01 * (df["revolving_utilization"] - 40)
    - 0.00001 * (df["annual_income"] - 60000)
    + 0.5 * (df["term_months"] == 60).astype(int)
    + 0.3 * (df["home_ownership"] == "RENT").astype(int)
    - 0.02 * df["employment_length"]
    - 0.4 * (df["months_since_last_delinquency"] == 999).astype(int)
    + np.random.normal(0, 0.5, n_samples)
)

default_prob = 1 / (1 + np.exp(-default_logit))
df["default"] = (np.random.random(n_samples) < default_prob).astype(int)

print(f"   Samples: {len(df)}")
print(f"   Default rate: {df['default'].mean():.1%}")

# Save raw data
df.to_csv("data/credit_data.csv", index=False)

# ──────────────────────────────────────────────
# 2. Feature Engineering & Preprocessing
# ──────────────────────────────────────────────
print("\n⚙️  Preprocessing features...")

# Encode categoricals
le_home = LabelEncoder()
le_purpose = LabelEncoder()
df["home_ownership_enc"] = le_home.fit_transform(df["home_ownership"])
df["loan_purpose_enc"] = le_purpose.fit_transform(df["loan_purpose"])

# Engineered features
df["loan_to_income"] = np.round(df["loan_amount"] / df["annual_income"], 4)
df["credit_utilization_score"] = np.round(df["revolving_utilization"] * df["num_open_accounts"] / 100, 2)
df["has_delinquency_history"] = (df["months_since_last_delinquency"] < 999).astype(int)

feature_cols = [
    "loan_amount", "annual_income", "interest_rate", "dti_ratio", "credit_score",
    "employment_length", "num_open_accounts", "num_derogatory_records",
    "revolving_utilization", "total_credit_lines", "months_since_last_delinquency",
    "home_ownership_enc", "loan_purpose_enc", "term_months",
    "loan_to_income", "credit_utilization_score", "has_delinquency_history"
]

# Pretty names for SHAP display
feature_display_names = {
    "loan_amount": "Loan Amount ($)",
    "annual_income": "Annual Income ($)",
    "interest_rate": "Interest Rate (%)",
    "dti_ratio": "Debt-to-Income Ratio (%)",
    "credit_score": "Credit Score",
    "employment_length": "Employment Length (years)",
    "num_open_accounts": "Open Accounts",
    "num_derogatory_records": "Derogatory Records",
    "revolving_utilization": "Revolving Utilization (%)",
    "total_credit_lines": "Total Credit Lines",
    "months_since_last_delinquency": "Months Since Last Delinquency",
    "home_ownership_enc": "Home Ownership",
    "loan_purpose_enc": "Loan Purpose",
    "term_months": "Loan Term (months)",
    "loan_to_income": "Loan-to-Income Ratio",
    "credit_utilization_score": "Credit Utilization Score",
    "has_delinquency_history": "Has Delinquency History"
}

X = df[feature_cols]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ──────────────────────────────────────────────
# 3. Train XGBoost Model
# ──────────────────────────────────────────────
print("\n🤖 Training XGBoost model...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📈 Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Good Loan", "Default"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ──────────────────────────────────────────────
# 4. Generate SHAP Explainer
# ──────────────────────────────────────────────
print("\n🔍 Generating SHAP explainer...")

explainer = shap.TreeExplainer(model)
# Pre-compute SHAP values for test set (for global plots)
shap_values_test = explainer.shap_values(X_test)

# ──────────────────────────────────────────────
# 5. Save All Artifacts
# ──────────────────────────────────────────────
print("\n💾 Saving model artifacts...")

joblib.dump(model, "models/xgb_credit_risk_model.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")
joblib.dump(le_home, "models/le_home.pkl")
joblib.dump(le_purpose, "models/le_purpose.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
joblib.dump(feature_display_names, "models/feature_display_names.pkl")
joblib.dump(shap_values_test, "models/shap_values_test.pkl")
joblib.dump(X_test, "models/X_test.pkl")

# Save metadata
metadata = {
    "n_samples": n_samples,
    "n_features": len(feature_cols),
    "default_rate": float(df["default"].mean()),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "home_ownership_classes": list(le_home.classes_),
    "loan_purpose_classes": list(le_purpose.classes_),
    "feature_ranges": {
        col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
        for col in ["loan_amount", "annual_income", "interest_rate", "dti_ratio", "credit_score",
                     "employment_length", "num_open_accounts", "num_derogatory_records",
                     "revolving_utilization", "total_credit_lines"]
    }
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ All artifacts saved to models/")
print("   Ready to run: streamlit run app.py")
