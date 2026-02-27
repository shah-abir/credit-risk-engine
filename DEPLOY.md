# 🚀 Deployment Guide — Credit Risk Scoring Engine

Follow these steps to get your app live on Streamlit Cloud with a shareable URL.

---

## Step 1: Push to GitHub

```bash
# Navigate to the project folder
cd credit-risk-engine

# Initialize git
git init
git add .
git commit -m "Initial commit: Credit Risk Scoring Engine with Explainable AI"

# Create a new repo on GitHub (go to github.com/new)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-engine.git
git branch -M main
git push -u origin main
```

**Important:** Make sure the `models/` folder with all `.pkl` files is committed.
The model files are needed for the app to run.

---

## Step 2: Deploy on Streamlit Cloud (Free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Select:
   - **Repository**: `YOUR_USERNAME/credit-risk-engine`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**
6. Wait 2-3 minutes for it to build and launch

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## Step 3: Add the Live URL to Your Resume & GitHub

1. Update `README.md` — replace the demo link with your actual Streamlit URL
2. Add the link to your resume under the project description
3. Add it to your LinkedIn Featured section
4. Pin the GitHub repo on your profile

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App crashes on load | Check that all files in `models/` are committed to GitHub |
| SHAP plots not rendering | Ensure `matplotlib` is in `requirements.txt` |
| Slow first load | Normal — Streamlit Cloud cold-starts take 30-60 seconds |
| "Module not found" | Check `requirements.txt` has all packages listed |

---

## Optional: Custom Domain

Streamlit Cloud free tier gives you a `.streamlit.app` URL.
For a custom domain, you'd need Streamlit Teams ($) or deploy on Railway/Render instead.

---

That's it! You should have a live, shareable credit risk app within 10 minutes.
