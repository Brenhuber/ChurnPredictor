import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------- Streamlit App Appearance --------------------

st.set_page_config(
    page_title="Modern Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #354f52 0%, #2f3e46 100%) !important;
    }
    .stSidebar {
        background-color: #2f3e46;
        color: #ffffff;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput {
        color: #000000;
    }
    .stButton>button {
        background-color: #52796f !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #354f52 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Load and Preprocess Data ------------------

df = pd.read_csv("Telco-Customer-Churn.csv")
df = df.drop('customerID', axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df)

# -------------------- Sidebar --------------------

st.sidebar.title("Customer Details")
customer = {
    "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.sidebar.selectbox("Senior Citizen?", [0, 1]),
    "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
    "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "OnlineBackup": st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "DeviceProtection": st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "TechSupport": st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "StreamingTV": st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    "MonthlyCharges": st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0),
    "TotalCharges": st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0),
}

# ---------------------- Model Selection --------------------

models = {
    "Logistic Regression (Preferred Model)": LogisticRegression(random_state=42, max_iter=10000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=9),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
}

selected_model = st.sidebar.radio("Select Model Type", list(models.keys()))

# ---------------------- Prediction Function --------------------

def predict(customer_dict, df, model):
    single = pd.DataFrame([customer_dict])
    single = pd.get_dummies(single).reindex(columns=df.drop("Churn", axis=1).columns, fill_value=0)
    X = pd.get_dummies(df.drop("Churn", axis=1))
    y = df["Churn"]
    X = X.reindex(columns=X.columns, fill_value=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    proba = model.predict_proba(single[X_train.columns])[0][1]

    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X_train.columns)
    elif hasattr(model, "coef_"):
        fi = pd.Series(np.abs(model.coef_[0]), index=X_train.columns)
    else:
        fi = None

    if fi is not None:
        fi = fi.sort_values(ascending=False).head(10)

    return acc, proba, fi

# ---------------------- Main App Logic --------------------

if st.sidebar.button("Predict"):
    with st.spinner("Running model..."):
        accuracy, churn_proba, fi = predict(customer, df, models[selected_model])
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Churn Probability", f"{churn_proba*100:.1f}%")
    if churn_proba > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is unlikely to churn.")

    # Gauge Chart for Churn Probability
    st.subheader("Churn Probability Gauge")
    fig1 = go.Figure(go.Indicator(mode="gauge+number", value=churn_proba, gauge={'axis': {'range': [0, 1]}}))
    st.plotly_chart(fig1, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance")
    if fi is not None and not fi.empty:
        fig = px.bar(x=fi.values, y=fi.index, orientation='h', labels={'x': 'Importance', 'y': 'Feature'})
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
        fig.update_traces(marker_color='#52796f')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            background: #ffffff22;
            border: 2px solid #52796f;
            border-radius: 16px;
            margin: 2rem 0;
        ">
            <h1 style="
                color: #ffffff;
                font-size: 2rem;
                font-weight: 600;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                text-align: center;
                line-height: 1.3;
            ">
                🔍 Complete the form in the sidebar,<br>
                select a model, and click <strong>Predict</strong><br>
                to view churn insights.
            </h1>
        </div>
    """, unsafe_allow_html=True)
