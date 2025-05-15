# Diabetes Prediction API

A FastAPI-based machine learning API for diabetes prediction. **Optimized for Vercel deployment**.

---

## 🚨 Critical Pre-Deployment Notes

**Before proceeding, ensure:**
1. **Python 3.9.6** is used (Vercel fails with 3.12)
2. Model file (`*.sav`) <50MB (Vercel serverless limit)
3. `scikit-learn==1.2.2` and `numpy==1.23.5` in requirements

---

## 📦 Project Structure
├── main.py # FastAPI application
├── diabetes_model.sav # ML model (MUST be <50MB)
├── requirements.txt # Python dependencies
├── runtime.txt # Python version spec
├── vercel.json # Vercel config
└── README.md


---

## ⚙️ Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-api.git
   cd diabetes-api

   python3.9 -m venv venv  # Must use Python 3.9
source venv/bin/activate

pip install -r requirements.txt --no-cache-dir
