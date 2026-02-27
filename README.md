# 🏦 Credit Risk Scoring Engine with Explainable AI

An interactive credit risk assessment tool powered by **XGBoost** with **SHAP-based explainability**, demonstrating production-ready ML for financial services.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.46-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)

## 🔗 [Live Demo →](https://credit-risk-engine-fpnuuevmfav5fgyuw96whv.streamlit.app/)

## What It Does

Users adjust 14 loan parameters (income, credit score, DTI, etc.) via sliders and instantly see:
- **Risk score** with color-coded gauge (Low / Medium / High)
- **SHAP waterfall plots** explaining *why* the model made that decision
- **Top risk factors** ranked by impact with plain-language summaries
- **Global insights** showing feature importance across all predictions

## Why It Matters

Regulations like **ECOA** and **GDPR** require lenders to explain credit decisions. This project shows how SHAP makes black-box ML models transparent and audit-ready.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost (10K records, 17 features, ROC AUC ~0.77) |
| Explainability | SHAP (TreeExplainer — waterfall, force, beeswarm plots) |
| Frontend | Streamlit + Plotly |
| Data | Pandas, NumPy, Scikit-learn |

## Quick Start

```bash
git clone https://github.com/shah-abir/credit-risk-engine.git
cd credit-risk-engine
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model: `python train_model.py`

## Author

**Shah Md Abir Hussain** — MS Financial Technology, University of Connecticut
[LinkedIn](https://linkedin.com/in/shah-abir) · [Email](mailto:Abir.h.shah@gmail.com)
