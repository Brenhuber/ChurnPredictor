<h1 align="center">📊 Modern Churn Predictor</h1>
<p align="center"><em>Predict customer churn with interactive insights and feature importances</em></p>

---

### 🚀 Overview

**Modern Churn Predictor** is a Streamlit-based app that predicts telco customer churn. It offers multiple models (Logistic Regression, Random Forest, Decision Tree, Gradient Boosting), displays prediction probability with a gauge chart, and surfaces top feature importances.

---

### ✨ Features

- Input customer profile via sidebar widgets  
- Select from four ML models with a single click  
- Display model accuracy and churn probability  
- Interactive gauge chart for churn likelihood  
- Top-10 feature importance bar chart  
- Modern, responsive UI with custom CSS  

---

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![pandas](https://img.shields.io/badge/pandas-Data%20Handling-purple?logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn) ![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-orange?logo=plotly)

---

### ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brenhuber/ChurnPredictor.git
   cd ChurnPredictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or with conda:
   # conda install -c conda-forge streamlit pandas scikit-learn plotly numpy
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   
---

### 🧭 Usage

- Fill in customer details in the sidebar.
- Choose a model (Logistic Regression, Random Forest, Decision Tree, Gradient Boosting).
- Click Predict to view accuracy, churn probability gauge, and feature importance.

---

### 📋 Requirements

- Python 3.8–3.11
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
