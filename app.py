"""
🏦 Credit Risk Scoring Engine with Explainable AI
Built by Shah Md Abir Hussain
─────────────────────────────────────────────────
Interactive credit risk assessment tool powered by XGBoost
with SHAP-based explainability for transparent decision-making.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B3A5C;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 1.5rem;
    }
    .risk-score-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1B3A5C;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background: #f0f4f8;
    }
    .factor-positive {
        color: #28a745;
        font-weight: 600;
    }
    .factor-negative {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Model Artifacts
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgb_credit_risk_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    le_home = joblib.load("models/le_home.pkl")
    le_purpose = joblib.load("models/le_purpose.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    display_names = joblib.load("models/feature_display_names.pkl")
    shap_values_test = joblib.load("models/shap_values_test.pkl")
    X_test = joblib.load("models/X_test.pkl")
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    return model, explainer, le_home, le_purpose, feature_cols, display_names, shap_values_test, X_test, metadata

model, explainer, le_home, le_purpose, feature_cols, display_names, shap_values_test, X_test, metadata = load_artifacts()

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 Credit Risk Scoring Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered credit risk assessment with explainable predictions &nbsp;|&nbsp; Built by <b>Shah Md Abir Hussain</b></p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Loan Application Input
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Loan Application Details")
    st.markdown("---")

    st.markdown("**Loan Information**")
    loan_amount = st.slider("Loan Amount ($)", 1000, 40000, 15000, step=500)
    interest_rate = st.slider("Interest Rate (%)", 5.0, 28.0, 12.0, step=0.25)
    term_months = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months ({x//12} years)")
    loan_purpose = st.selectbox("Loan Purpose", sorted(metadata["loan_purpose_classes"]),
                                 format_func=lambda x: x.replace("_", " ").title())

    st.markdown("---")
    st.markdown("**Borrower Profile**")
    annual_income = st.slider("Annual Income ($)", 15000, 300000, 65000, step=5000)
    credit_score = st.slider("Credit Score", 300, 850, 700, step=5)
    employment_length = st.slider("Employment Length (years)", 0, 10, 5)
    home_ownership = st.selectbox("Home Ownership", sorted(metadata["home_ownership_classes"]))

    st.markdown("---")
    st.markdown("**Credit History**")
    dti_ratio = st.slider("Debt-to-Income Ratio (%)", 0.0, 45.0, 18.0, step=0.5)
    revolving_utilization = st.slider("Revolving Utilization (%)", 0.0, 100.0, 35.0, step=1.0)
    num_open_accounts = st.slider("Open Credit Accounts", 1, 40, 10)
    total_credit_lines = st.slider("Total Credit Lines", 2, 60, 22)
    num_derogatory_records = st.slider("Derogatory Records", 0, 5, 0)
    months_delinquency = st.selectbox(
        "Months Since Last Delinquency",
        [999, 72, 60, 48, 36, 24, 18, 12, 6, 0],
        format_func=lambda x: "Never" if x == 999 else f"{x} months ago"
    )

    analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")


# ──────────────────────────────────────────────
# Prepare Input & Predict
# ──────────────────────────────────────────────
def prepare_input():
    home_enc = le_home.transform([home_ownership])[0]
    purpose_enc = le_purpose.transform([loan_purpose])[0]
    loan_to_income = round(loan_amount / annual_income, 4)
    credit_util_score = round(revolving_utilization * num_open_accounts / 100, 2)
    has_delinquency = 1 if months_delinquency < 999 else 0

    input_data = pd.DataFrame([{
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "interest_rate": interest_rate,
        "dti_ratio": dti_ratio,
        "credit_score": credit_score,
        "employment_length": employment_length,
        "num_open_accounts": num_open_accounts,
        "num_derogatory_records": num_derogatory_records,
        "revolving_utilization": revolving_utilization,
        "total_credit_lines": total_credit_lines,
        "months_since_last_delinquency": months_delinquency,
        "home_ownership_enc": home_enc,
        "loan_purpose_enc": purpose_enc,
        "term_months": term_months,
        "loan_to_income": loan_to_income,
        "credit_utilization_score": credit_util_score,
        "has_delinquency_history": has_delinquency
    }])

    return input_data[feature_cols]

input_df = prepare_input()
risk_prob = model.predict_proba(input_df)[0][1]
risk_score = int(risk_prob * 1000)
shap_values_single = explainer.shap_values(input_df)

# ──────────────────────────────────────────────
# Determine Risk Level
# ──────────────────────────────────────────────
if risk_prob < 0.15:
    risk_level = "LOW RISK"
    risk_class = "risk-low"
    risk_emoji = "✅"
    risk_color = "#28a745"
    decision = "APPROVE"
elif risk_prob < 0.40:
    risk_level = "MEDIUM RISK"
    risk_class = "risk-medium"
    risk_emoji = "⚠️"
    risk_color = "#ffc107"
    decision = "REVIEW"
else:
    risk_level = "HIGH RISK"
    risk_class = "risk-high"
    risk_emoji = "🚨"
    risk_color = "#dc3545"
    decision = "DECLINE"

# ──────────────────────────────────────────────
# Main Content — Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Assessment", "🔍 SHAP Explanation", "📈 Global Insights", "ℹ️ About"])

# ═══════════════════════════════════════════════
# TAB 1: Risk Assessment
# ═══════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="risk-score-box {risk_class}">
            <div style="font-size: 3rem; font-weight: 800;">{risk_emoji} {risk_score}</div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.3rem;">{risk_level}</div>
            <div style="font-size: 0.85rem; color: #555; margin-top: 0.3rem;">Default Probability: {risk_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #666;">Recommended Decision</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {risk_color};">{decision}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 15], "color": "#d4edda"},
                    {"range": [15, 40], "color": "#fff3cd"},
                    {"range": [40, 100], "color": "#f8d7da"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": risk_prob * 100
                }
            },
            title={"text": "Default Probability", "font": {"size": 14}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Loan Amount</div>
            <div style="font-size: 1.2rem; font-weight: 700;">${loan_amount:,}</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Annual Income</div>
            <div style="font-size: 1.2rem; font-weight: 700;">${annual_income:,}</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Credit Score</div>
            <div style="font-size: 1.2rem; font-weight: 700;">{credit_score}</div>
        </div>
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Interest Rate</div>
            <div style="font-size: 1.2rem; font-weight: 700;">{interest_rate}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Top factors
    st.markdown("### 🔑 Key Risk Factors")
    shap_df = pd.DataFrame({
        "Feature": [display_names.get(f, f) for f in feature_cols],
        "SHAP Value": shap_values_single[0],
        "Direction": ["Increases Risk" if v > 0 else "Decreases Risk" for v in shap_values_single[0]],
        "abs_shap": np.abs(shap_values_single[0])
    }).sort_values("abs_shap", ascending=False)

    top_factors = shap_df.head(8)

    fig_factors = go.Figure()
    colors = ["#dc3545" if v > 0 else "#28a745" for v in top_factors["SHAP Value"].values[::-1]]
    fig_factors.add_trace(go.Bar(
        x=top_factors["SHAP Value"].values[::-1],
        y=top_factors["Feature"].values[::-1],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top_factors["SHAP Value"].values[::-1]],
        textposition="outside",
        textfont=dict(size=11)
    ))
    fig_factors.update_layout(
        title="Top 8 Factors Influencing This Prediction",
        xaxis_title="SHAP Value (Impact on Default Probability)",
        yaxis_title="",
        height=350,
        margin=dict(t=40, b=40, l=10, r=40),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee", zeroline=True, zerolinecolor="#333", zerolinewidth=1.5)
    )
    st.plotly_chart(fig_factors, use_container_width=True)

    # Plain-language explanation
    st.markdown("### 💬 Plain-Language Explanation")
    top3_risk = shap_df[shap_df["SHAP Value"] > 0].head(3)
    top3_safe = shap_df[shap_df["SHAP Value"] < 0].head(3)

    explanation = "**Factors increasing risk:** "
    if len(top3_risk) > 0:
        risk_items = [f"{row['Feature']} (impact: {row['SHAP Value']:+.3f})" for _, row in top3_risk.iterrows()]
        explanation += ", ".join(risk_items) + ". "
    else:
        explanation += "None significant. "

    explanation += "\n\n**Factors decreasing risk:** "
    if len(top3_safe) > 0:
        safe_items = [f"{row['Feature']} (impact: {row['SHAP Value']:+.3f})" for _, row in top3_safe.iterrows()]
        explanation += ", ".join(safe_items) + "."
    else:
        explanation += "None significant."

    st.markdown(explanation)


# ═══════════════════════════════════════════════
# TAB 2: SHAP Explanation (Individual)
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 SHAP Waterfall — How This Prediction Was Made")
    st.markdown("Each bar shows how a single feature pushed the prediction from the base rate toward the final score.")

    # Waterfall plot
    fig_waterfall, ax = plt.subplots(figsize=(10, 6))
    shap_exp = shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=input_df.values[0],
        feature_names=[display_names.get(f, f) for f in feature_cols]
    )
    shap.plots.waterfall(shap_exp, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig_waterfall, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("### 📊 SHAP Force Plot")
    st.markdown("Visual breakdown of how each feature contributes to moving from the average prediction to this individual prediction.")

    # Force plot as matplotlib
    fig_force, ax = plt.subplots(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values_single[0],
        input_df.values[0],
        feature_names=[display_names.get(f, f) for f in feature_cols],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig_force, use_container_width=True)
    plt.close()

    # Full SHAP values table
    st.markdown("---")
    st.markdown("### 📋 Complete Feature Contribution Table")
    full_table = shap_df[["Feature", "SHAP Value", "Direction"]].copy()
    full_table["SHAP Value"] = full_table["SHAP Value"].map(lambda x: f"{x:+.4f}")
    st.dataframe(full_table.drop(columns=["abs_shap"] if "abs_shap" in full_table.columns else []),
                  use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 3: Global Insights
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("### 🌍 Global Feature Importance")
    st.markdown("Understanding which features matter most across all predictions in the test dataset.")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Global SHAP bar plot
        mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
        global_imp = pd.DataFrame({
            "Feature": [display_names.get(f, f) for f in feature_cols],
            "Mean |SHAP|": mean_abs_shap
        }).sort_values("Mean |SHAP|", ascending=True).tail(12)

        fig_global = go.Figure(go.Bar(
            x=global_imp["Mean |SHAP|"],
            y=global_imp["Feature"],
            orientation="h",
            marker_color="#1B3A5C",
            text=[f"{v:.3f}" for v in global_imp["Mean |SHAP|"]],
            textposition="outside"
        ))
        fig_global.update_layout(
            title="Global Feature Importance (Mean |SHAP|)",
            height=450,
            margin=dict(t=40, b=30, l=10, r=40),
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee")
        )
        st.plotly_chart(fig_global, use_container_width=True)

    with col_g2:
        # SHAP summary (beeswarm) plot
        st.markdown("**SHAP Beeswarm Plot**")
        st.markdown("Each dot is a sample. Color = feature value. Position = SHAP impact.")
        fig_bee, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values_test,
            X_test,
            feature_names=[display_names.get(f, f) for f in feature_cols],
            max_display=12,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close()

    # Model performance
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("ROC AUC Score", f"{metadata['roc_auc']:.4f}")
    with perf_col2:
        st.metric("Training Samples", f"{metadata['n_samples']:,}")
    with perf_col3:
        st.metric("Features Used", f"{metadata['n_features']}")


# ═══════════════════════════════════════════════
# TAB 4: About
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("""
    ### About This Project

    **Credit Risk Scoring Engine with Explainable AI** is a portfolio project demonstrating
    end-to-end machine learning pipeline development for financial risk assessment.

    #### Technical Architecture
    - **Model**: XGBoost gradient-boosted classifier trained on 10,000 synthetic credit records
    - **Explainability**: SHAP (SHapley Additive exPlanations) for both local and global interpretability
    - **Features**: 17 engineered features including loan-to-income ratio, credit utilization score, and delinquency history
    - **Frontend**: Streamlit with Plotly interactive visualizations
    - **Deployment**: Streamlit Cloud (free tier)

    #### Why Explainable AI Matters in Finance
    In regulated industries like banking and lending, it is not enough for a model to be accurate —
    it must be **interpretable**. Regulatory frameworks such as the Equal Credit Opportunity Act (ECOA)
    and GDPR require that lenders can explain why a credit application was denied. SHAP provides
    mathematically grounded explanations for each individual prediction, making AI decisions
    transparent, auditable, and fair.

    #### Key Features
    - **Real-time risk scoring** with interactive parameter adjustment
    - **Individual explanations** via SHAP waterfall and force plots
    - **Global insights** via beeswarm plots and feature importance rankings
    - **Plain-language explanations** translating model output to human-readable factors
    - **Production-ready architecture** with cached model loading and modular design

    #### Tech Stack
    `Python` · `XGBoost` · `SHAP` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `Scikit-learn`

    ---
    **Built by Shah Md Abir Hussain**
    MS Financial Technology, University of Connecticut
    [GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/)
    """)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.8rem;">'
    'Credit Risk Scoring Engine v1.0 &nbsp;|&nbsp; Built with XGBoost + SHAP + Streamlit &nbsp;|&nbsp; '
    'Shah Md Abir Hussain &nbsp;|&nbsp; 2026'
    '</div>',
    unsafe_allow_html=True
)
