# 🏦 Credit Risk Scoring Engine with Explainable AI

An interactive credit risk assessment tool powered by **XGBoost** with **SHAP-based explainability**, built to demonstrate production-ready ML pipeline development for financial services.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.46-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)

## 🔗 Live Demo

👉 **[Launch App on Streamlit Cloud](https://credit-risk-engine.streamlit.app)** *(update with your actual URL after deployment)*

## 📸 Screenshots

| Risk Assessment | SHAP Explanation | Global Insights |
|---|---|---|
| Interactive scoring with real-time gauge | Waterfall & force plots | Beeswarm & importance charts |

## 🎯 Why This Project?

In regulated financial industries, **model interpretability is not optional** — it's required by law. Frameworks like ECOA (Equal Credit Opportunity Act) and GDPR mandate that lenders must explain credit decisions to applicants. This project demonstrates:

- **End-to-end ML pipeline**: Data generation → Feature engineering → Model training → Deployment
- **Explainable AI (XAI)**: SHAP values provide mathematically grounded explanations for every prediction
- **Production-ready architecture**: Cached model loading, modular design, interactive UI
- **Financial domain expertise**: Credit risk features, loan assessment logic, regulatory awareness

## 🏗️ Architecture

```
credit-risk-engine/
├── app.py                    # Streamlit application (main entry point)
├── train_model.py            # Model training pipeline
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── models/
│   ├── xgb_credit_risk_model.pkl    # Trained XGBoost model
│   ├── shap_explainer.pkl           # SHAP TreeExplainer
│   ├── le_home.pkl                  # Home ownership encoder
│   ├── le_purpose.pkl               # Loan purpose encoder
│   ├── feature_cols.pkl             # Feature column names
│   ├── feature_display_names.pkl    # Human-readable feature names
│   ├── shap_values_test.pkl         # Pre-computed test set SHAP values
│   ├── X_test.pkl                   # Test set features
│   └── metadata.json               # Model metadata & feature ranges
└── data/
    └── credit_data.csv              # Generated training dataset
```

## 🧠 Technical Details

### Model
- **Algorithm**: XGBoost (Gradient-Boosted Decision Trees)
- **Training Data**: 10,000 synthetic credit records with realistic feature distributions
- **Features**: 17 features including 3 engineered (loan-to-income ratio, credit utilization score, delinquency flag)
- **Performance**: ROC AUC ~0.77 on held-out test set
- **Class Balancing**: Scale-adjusted positive weights for imbalanced default rate (~7%)

### Explainability
- **SHAP (SHapley Additive exPlanations)**: Game-theory based approach to explain predictions
- **Local explanations**: Waterfall plots, force plots, and feature contribution tables for individual predictions
- **Global explanations**: Beeswarm plots and mean absolute SHAP importance across the test set
- **Plain-language summaries**: Auto-generated human-readable risk factor explanations

### Features Used
| Category | Features |
|----------|----------|
| Loan Info | Amount, Interest Rate, Term, Purpose |
| Borrower | Income, Credit Score, Employment Length, Home Ownership |
| Credit History | DTI Ratio, Revolving Utilization, Open Accounts, Derogatory Records, Delinquency History |
| Engineered | Loan-to-Income Ratio, Credit Utilization Score, Has Delinquency Flag |

## 🚀 Quick Start

### Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-engine.git
cd credit-risk-engine

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python train_model.py

# Launch the app
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — done! 🎉

## 📊 Key App Features

- **🎛️ Interactive Risk Scoring**: Adjust 14 loan parameters via sidebar sliders and see risk score update in real-time
- **📊 Risk Gauge**: Visual probability gauge with color-coded risk levels (Low/Medium/High)
- **🔍 SHAP Waterfall Plot**: See exactly how each feature pushed the prediction from the base rate
- **💪 Force Plot**: Visualize the tug-of-war between risk-increasing and risk-decreasing factors
- **🌍 Global Insights**: Understand which features matter most across all predictions
- **💬 Plain-Language Explanations**: Auto-generated human-readable risk summaries
- **📋 Full Contribution Table**: Complete SHAP values for all 17 features

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost |
| Explainability | SHAP |
| Frontend | Streamlit |
| Visualization | Plotly, Matplotlib |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Deployment | Streamlit Cloud |

## 📄 License

MIT License — free to use, modify, and distribute.

## 👤 Author

**Shah Md Abir Hussain**
MS Financial Technology, University of Connecticut
- GitHub: [github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [linkedin.com/in/YOUR_PROFILE](https://linkedin.com/in/YOUR_PROFILE)
- Email: Abir.h.shah@gmail.com
