import pickle
import numpy as np
import streamlit as st

# १. पेज लेआउट आणि टायटल सेट करणे
st.set_page_config(
    page_title="Placement Predictor", page_icon="🎓", layout="centered"
)

# २. आकर्षक डिझाइनसाठी Custom CSS (UI Polish)
st.markdown(
    """
    <style>
    /* मुख्य बॅकग्राउंड हलका ग्रे रंग */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* टायटल स्टाइल */
    .main-title {
        color: #1e3a8a;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #4b5563;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Predict Button चे स्टाइल */
    div.stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.6rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.3);
    }
    
    div.stButton > button:hover {
        background-color: #e03e3e;
        color: white;
        transform: translateY(-2px);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ३. Perceptron Model (.pkl file) लोड करणे
@st.cache_resource
def load_model():
    # तुमच्या मॉडेल फाइलचे नाव 'perceptron.pkl' ठेवा
    with open("perceptron.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as e:
    st.error(
        "⚠️ 'perceptron.pkl' ही मॉडेल फाइल सापडली नाही. कृपया ती याच फोल्डरमध्ये ठेवा."
    )
    st.stop()

# ४. हेडिंग आणि डिस्क्रिप्शन
st.markdown(
    "<h1 class='main-title'>🎓 Student Placement Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>तुमचा <b>CGPA</b> आणि <b>Resume Score</b> टाकून सिलेक्ट होण्याची शक्यता तपासा.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ५. इनपुट कार्ड लेआउट (दोन कॉलम्स)
col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "📊 CGPA (0.0 to 10.0)",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
        help="तुमचा CGPA प्रविष्ट करा",
    )

with col2:
    resume_score = st.number_input(
        "📄 Resume Score (0.0 to 10.0)",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1,
        help="तुमचा Resume Score प्रविष्ट करा",
    )

st.write("")  # Space

# ६. Predict Button Logic
if st.button("🚀 Predict Placement"):
    # प्रेडिक्ट बटण क्लिक झाल्यावर फुगे/बॅलन्स येतील
    st.balloons()

    # Model साठी Input Array तयार करणे
    input_data = np.array([[cgpa, resume_score]])

    # प्रेडिक्शन
    prediction = model.predict(input_data)

    st.markdown("---")

    # निकाल दाखवणे
    if prediction[0] == 1:
        st.success("🎉 **अभिनंदन! तुम्ही प्लेसमेंटसाठी सिलेक्ट होऊ शकता.**")
    else:
        st.error(
            "⚠️ **सध्या सिलेक्ट होण्याची शक्यता कमी आहे. CGPA किंवा Resume Score वाढवण्याचा प्रयत्न करा.**"
        )
