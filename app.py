import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Placement Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Dark UI matching the second image
st.markdown(
    """
    <style>
    /* Dark Background */
    .stApp {
        background-color: #0d1117;
        color: #ffffff;
    }
    
    /* Center Card Container */
    .block-container {
        max-width: 480px !important;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Card Design */
    div[data-testid="stVerticalBlock"] > div:has(div.card-marker) {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.5);
    }

    /* Headings */
    .main-title {
        color: #ffffff;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .sub-title {
        color: #8b949e;
        text-align: center;
        font-size: 13px;
        margin-bottom: 25px;
    }

    /* Inputs Labels */
    label {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Input Field Box Style */
    div[data-baseweb="input"] {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Input Field Hover & Focus */
    div[data-baseweb="input"]:focus-within {
        border-color: #8a2be2 !important;
    }

    /* Gradient Predict Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        margin-top: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0px 4px 15px rgba(168, 85, 247, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Trained Perceptron Model
@st.cache_resource
def load_model():
    with open("perceptron.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as e:
    st.error("Model file 'perceptron.pkl' not found!")
    st.stop()

# HTML Card Wrapper Start
st.markdown("<div class='card-marker'></div>", unsafe_allow_html=True)

# Title Section
st.markdown(
    "<div class='main-title'>Placement Predictor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>Perceptron Categorical Classifier</div>",
    unsafe_allow_html=True,
)

# Input Fields
cgpa = st.number_input(
    "CGPA (0 - 10)",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1,
    format="%.1f",
)

resume_score = st.number_input(
    "RESUME SCORE (0 - 10)",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.1,
    format="%.1f",
)

# Prediction Logic
if st.button("Predict Status"):
    # Trigger Balloons Animation
    st.balloons()

    # Model Input Array
    input_data = np.array([[cgpa, resume_score]])
    prediction = model.predict(input_data)

    st.markdown("<br>", unsafe_allow_html=True)

    # Output Result Display matching 2nd Image style
    if prediction[0] == 1:
        st.success("🎉 Category: Placed")
    else:
        st.error("⚠️ Category: Not Placed")
