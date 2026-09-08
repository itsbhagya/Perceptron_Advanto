import pickle
import numpy as np
import streamlit as st

# 1. Streamlit Page Config
st.set_page_config(
    page_title="Placement Predictor", page_icon="🎓", layout="centered"
)

# 2. Perfect Dark Theme Glassmorphism CSS
st.markdown(
    """
    <style>
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at center, #1b1b3a 0%, #0b0b18 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Native Header & Footer */
    header, footer, #MainMenu {
        visibility: hidden;
    }

    /* Limit container width */
    .block-container {
        max-width: 450px !important;
        padding-top: 4rem !important;
        padding-bottom: 2rem !important;
    }

    /* Form Container styling as single card */
    div[data-testid="stForm"] {
        background: rgba(26, 26, 48, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px 25px !important;
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(10px);
    }

    /* Header Styling */
    .card-header {
        text-align: center;
        margin-bottom: 25px;
    }

    .main-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }

    .sub-title {
        color: #7b7b9d;
        font-size: 13px;
        font-weight: 400;
    }

    /* Input Field Labels */
    .stNumberInput label {
        color: #8f8fae !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase;
        margin-bottom: 6px !important;
    }

    /* Dark Input Box Styling */
    div[data-baseweb="input"] {
        background-color: #121225 !important;
        border: 1px solid #232342 !important;
        border-radius: 10px !important;
        padding: 2px 4px;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.3) !important;
    }

    input {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }

    /* Gradient Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
        padding: 12px;
        border-radius: 10px;
        border: none;
        margin-top: 10px;
        box-shadow: 0px 4px 15px rgba(168, 85, 247, 0.4);
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0px 6px 20px rgba(168, 85, 247, 0.6);
    }

    /* Custom Result Banner */
    .result-box-placed {
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid #22c55e;
        color: #4ade80;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        margin-top: 15px;
    }

    .result-box-not-placed {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        margin-top: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Model Loader
@st.cache_resource
def load_model():
    with open("perceptron.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception:
    st.error("Model file 'perceptron.pkl' not found!")
    st.stop()

# 4. Main Form Wrapper (Puts everything inside a single card)
with st.form("prediction_form", clear_on_submit=False):
    # Card Header
    st.markdown(
        """
        <div class="card-header">
            <div class="main-title">Placement Predictor</div>
            <div class="sub-title">Perceptron Categorical Classifier</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Input 1: CGPA
    cgpa = st.number_input(
        "CGPA (0 - 10)",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
        format="%.1f",
    )

    # Input 2: Resume Score
    resume_score = st.number_input(
        "RESUME SCORE (0 - 10)",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1,
        format="%.1f",
    )

    # Submit Button
    submitted = st.form_submit_button("Predict Status")

# 5. Prediction Logic and Display Output inside/below the Form State
if submitted:
    # Trigger Balloon Animation
    st.balloons()

    # Model Inference
    input_data = np.array([[cgpa, resume_score]])
    prediction = model.predict(input_data)

    # Show Output inside Card Style
    if prediction[0] == 1:
        st.markdown(
            '<div class="result-box-placed">🎉 Category: Placed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box-not-placed">⚠️ Category: Not Placed</div>',
            unsafe_allow_html=True,
        )

