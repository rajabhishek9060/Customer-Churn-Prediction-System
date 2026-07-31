# app.py
# Streamlit web application for Customer Churn Risk Prediction

import streamlit as st
import joblib
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Risk Alerts */
    .high-risk {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #fca5a5;
        text-align: center;
        margin-top: 1.5rem;
    }
    .low-risk {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #6ee7b7;
        text-align: center;
        margin-top: 1.5rem;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124,58,237,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")
FEATURE_IMP_PATH = os.path.join(PROJECT_ROOT, "artifacts", "feature_importance.json")

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the serialized Random Forest model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_importance():
    """Load feature importance JSON from disk."""
    if not os.path.exists(FEATURE_IMP_PATH):
        return {}
    with open(FEATURE_IMP_PATH, "r") as f:
        return json.load(f)


# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Powered by Random Forest · Telco Customer Intelligence</div>', unsafe_allow_html=True)
st.divider()

# ── Load Resources ─────────────────────────────────────────────────────────────
model = load_model()
feature_importance = load_feature_importance()

if model is None:
    st.error(
        "⚠️ Model not found. Please run `python src/data_prep.py` then `python src/train.py` first."
    )
    st.stop()

# Get the feature names the model was trained on
if hasattr(model, 'feature_names_in_'):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = list(feature_importance.keys()) if feature_importance else []

# ── Sidebar Form ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Customer Profile")
    st.markdown("*Configure the customer attributes below to predict churn risk.*")
    st.divider()

    # Core fields from specification
    tenure = st.slider(
        "📅 Tenure (months)",
        min_value=0, max_value=72, value=12, step=1,
        help="Number of months the customer has been with the company"
    )

    contract_options = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract_label = st.selectbox(
        "📄 Contract Type",
        options=list(contract_options.keys()),
        help="Type of contract the customer holds"
    )
    contract = contract_options[contract_label]

    payment_options = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    payment_label = st.selectbox(
        "💳 Payment Method",
        options=list(payment_options.keys()),
        help="Customer's preferred payment method"
    )
    payment = payment_options[payment_label]

    total_charges = st.number_input(
        "💰 Total Charges ($)",
        min_value=0.0, max_value=10000.0, value=500.0, step=10.0,
        help="Total amount charged to the customer"
    )

    monthly_charges = st.number_input(
        "🗓️ Monthly Charges ($)",
        min_value=0.0, max_value=200.0, value=65.0, step=1.0,
        help="Amount charged to the customer per month"
    )

    # Additional common Telco fields
    senior_citizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"])
    senior_citizen_val = 1 if senior_citizen == "Yes" else 0

    partner = st.selectbox("👥 Partner", ["No", "Yes"])
    partner_val = 1 if partner == "Yes" else 0

    dependents = st.selectbox("👨👩👧 Dependents", ["No", "Yes"])
    dependents_val = 1 if dependents == "Yes" else 0

    phone_service = st.selectbox("📞 Phone Service", ["No", "Yes"])
    phone_val = 1 if phone_service == "Yes" else 0

    internet_service_opts = {"DSL": 0, "Fiber optic": 1, "No": 2}
    internet_label = st.selectbox("🌐 Internet Service", list(internet_service_opts.keys()))
    internet_val = internet_service_opts[internet_label]

    online_security = st.selectbox("🔒 Online Security", ["No", "Yes", "No internet service"])
    online_security_val = {"No": 0, "Yes": 1, "No internet service": 2}[online_security]

    tech_support = st.selectbox("🛠️ Tech Support", ["No", "Yes", "No internet service"])
    tech_support_val = {"No": 0, "Yes": 1, "No internet service": 2}[tech_support]

    paperless_billing = st.selectbox("📧 Paperless Billing", ["No", "Yes"])
    paperless_val = 1 if paperless_billing == "Yes" else 0

    st.divider()
    predict_button = st.button("🔮 Predict Churn Risk", use_container_width=True)


# ── Main Content Area ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🎯 Prediction Result")

    if predict_button:
        # Build input feature vector matching model training columns
        # Map the sidebar inputs to a feature dict
        feature_map = {
            "tenure": tenure,
            "SeniorCitizen": senior_citizen_val,
            "Partner": partner_val,
            "Dependents": dependents_val,
            "PhoneService": phone_val,
            "MultipleLines": 0,
            "InternetService": internet_val,
            "OnlineSecurity": online_security_val,
            "OnlineBackup": 0,
            "DeviceProtection": 0,
            "TechSupport": tech_support_val,
            "StreamingTV": 0,
            "StreamingMovies": 0,
            "Contract": contract,
            "PaperlessBilling": paperless_val,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": 0,
        }

        # Build DataFrame with exactly the columns the model expects
        if feature_names:
            input_data = {}
            for feat in feature_names:
                input_data[feat] = feature_map.get(feat, 0)
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame([feature_map])

        # Run prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        churn_proba = proba[1] * 100

        # Display result
        if prediction == 1:
            st.markdown(
                f'<div class="high-risk">🔴 HIGH CHURN RISK<br><span style="font-size:0.9rem;font-weight:400;">This customer has a <strong>{churn_proba:.1f}%</strong> probability of churning.</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="low-risk">🟢 LOW CHURN RISK<br><span style="font-size:0.9rem;font-weight:400;">This customer has only a <strong>{churn_proba:.1f}%</strong> probability of churning.</span></div>',
                unsafe_allow_html=True
            )

        # Probability gauge
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Churn Probability",
            value=f"{churn_proba:.1f}%",
            delta=f"{churn_proba - 50:.1f}% vs baseline",
            delta_color="inverse"
        )
    else:
        st.info("👈 Configure the customer profile in the sidebar and click **Predict Churn Risk**.")


with col2:
    st.markdown("### 📈 Top 3 Feature Importances")

    if feature_importance:
        # Get top 3 features
        top_features = dict(list(feature_importance.items())[:3])
        labels = list(top_features.keys())
        values = list(top_features.values())

        # Create a stylish bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        colors = ["#a78bfa", "#60a5fa", "#34d399"]
        bars = ax.barh(labels[::-1], values[::-1], color=colors, height=0.5, edgecolor="none")

        # Value labels on bars
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color="#e0e0e0", fontsize=11, fontweight="600"
            )

        ax.set_xlabel("Importance Score", color="#9ca3af", fontsize=10)
        ax.set_title("Model Feature Drivers", color="#e0e0e0", fontsize=12, fontweight="700", pad=12)
        ax.tick_params(colors="#9ca3af")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#374151")
        ax.spines["left"].set_color("#374151")
        ax.set_xlim(0, max(values) * 1.25)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("Feature importance data not found. Run `python src/train.py` first.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "Customer Churn Prediction System · Built with ❤️ using Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
