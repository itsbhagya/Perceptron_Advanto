import os
import pickle
import numpy as np
import streamlit as st
from sklearn.linear_model import Perceptron

# ==============================================================================
# 1. Automatic Model Creation Logic (Runs if model.pkl does not exist)
# ==============================================================================
MODEL_FILE = "model.pkl"

if not os.path.exists(MODEL_FILE):
    # Training sample data (CGPA and Resume Score)
    # Features: ['cgpa', 'resume_score']
    X_train = np.array([
        [5.0, 50],
        [6.0, 60],
        [6.5, 65],
        [7.0, 70],
        [7.8, 80],
        [8.5, 85],
        [9.0, 90],
        [9.5, 95]
    ])
    # Labels: 0 = Not Placed, 1 = Placed
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Perceptron Model Configuration matching standard scikit-learn settings
    model = Perceptron(max_iter=1000, random_state=0)
    model.fit(X_train, y_train)

    # Save model as model.pkl
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# ==============================================================================
# 2. Streamlit Web Application Interface
# ==============================================================================
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #00c853;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00e676;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🎓 Candidate Placement Predictor")
st.write("Aapne `CGPA` aur `Resume Score` ke basis par model ka outcome predict karein.")

st.divider()

# Load Saved Model Function
@st.cache_resource
def get_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

perceptron_model = get_model()

# Input UI Fields
col1, col2 = st.columns(2)

with col1:
    cgpa_input = st.number_input(
        "Enter CGPA (0.0 - 10.0)",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

with col2:
    resume_input = st.number_input(
        "Enter Resume Score (0 - 100)",
        min_value=0,
        max_value=100,
        value=75,
        step=1
    )

st.write("")  # Spacing

# Prediction Button & Logic
if st.button("🔮 Check Placement Status"):
    # Reshape input data for Scikit-Learn
    features = np.array([[cgpa_input, resume_input]])
    
    # Predict output
    result = perceptron_model.predict(features)[0]
    
    st.divider()
    
    if result == 1:
        st.balloons()
        st.success("🎉 **Selection Chances High:** Candidate Placement ke liye eligible hai!")
    else:
        st.error("⚠️ **Selection Chances Low:** Candidate ko CGPA / Resume Score improve karne ki zaroorat hai.")

st.caption("Model Architecture: Scikit-Learn Perceptron Classifier")
















