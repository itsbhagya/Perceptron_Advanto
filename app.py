from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
MODEL_PATH = "perceptron.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ---------------------------------------------------------
# HTML + CSS
# ---------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Placement Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, #312e81 0%, transparent 35%),
                radial-gradient(circle at bottom right, #581c87 0%, transparent 35%),
                linear-gradient(135deg, #090b1a, #11142b, #17132d);

            color: white;

            display: flex;
            justify-content: center;
            align-items: center;

            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 600px;
        }

        .card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);

            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);

            border-radius: 25px;

            padding: 35px;

            box-shadow:
                0 25px 70px rgba(0, 0, 0, 0.45);

            animation: fadeIn 0.7s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .badge {
            display: inline-block;

            padding: 7px 14px;

            border-radius: 30px;

            background: rgba(139, 92, 246, 0.18);

            color: #c4b5fd;

            font-size: 13px;

            margin-bottom: 15px;
        }

        h1 {
            font-size: 38px;
            margin-bottom: 10px;

            background: linear-gradient(
                90deg,
                #c4b5fd,
                #a78bfa,
                #f0abfc
            );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: #b8b9ca;
            font-size: 15px;
            line-height: 1.6;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 22px;
        }

        label {
            display: block;

            margin-bottom: 9px;

            color: #e5e7eb;

            font-size: 14px;
            font-weight: 600;
        }

        input {
            width: 100%;

            padding: 15px 17px;

            border-radius: 12px;

            border: 1px solid rgba(255,255,255,0.15);

            background: rgba(0,0,0,0.25);

            color: white;

            font-size: 16px;

            outline: none;

            transition: 0.3s;
        }

        input:focus {
            border-color: #a78bfa;

            box-shadow:
                0 0 0 3px rgba(167,139,250,0.12);
        }

        input::placeholder {
            color: #777b91;
        }

        .hint {
            font-size: 12px;
            color: #888ca2;
            margin-top: 6px;
        }

        .predict-btn {
            width: 100%;

            padding: 16px;

            border: none;

            border-radius: 13px;

            background:
                linear-gradient(
                    135deg,
                    #7c3aed,
                    #a855f7,
                    #ec4899
                );

            color: white;

            font-size: 17px;

            font-weight: bold;

            cursor: pointer;

            transition: all 0.3s;

            box-shadow:
                0 10px 30px rgba(124,58,237,0.3);
        }

        .predict-btn:hover {
            transform: translateY(-2px);

            box-shadow:
                0 15px 35px rgba(168,85,247,0.4);
        }

        .predict-btn:active {
            transform: scale(0.98);
        }

        .result {
            margin-top: 28px;

            padding: 24px;

            border-radius: 18px;

            background: rgba(255,255,255,0.07);

            border: 1px solid rgba(255,255,255,0.12);

            text-align: center;

            animation: resultIn 0.5s ease;
        }

        @keyframes resultIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }

            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        .result-title {
            color: #aeb0c5;

            font-size: 13px;

            margin-bottom: 8px;

            text-transform: uppercase;

            letter-spacing: 1px;
        }

        .placed {
            color: #86efac;

            font-size: 32px;

            font-weight: bold;
        }

        .not-placed {
            color: #fca5a5;

            font-size: 32px;

            font-weight: bold;
        }

        .decision {
            margin-top: 15px;

            color: #c4c6d8;

            font-size: 14px;
        }

        .confidence-box {
            margin-top: 20px;

            text-align: left;
        }

        .confidence-header {
            display: flex;

            justify-content: space-between;

            color: #bfc1d3;

            font-size: 13px;

            margin-bottom: 8px;
        }

        .progress {
            height: 8px;

            background: rgba(255,255,255,0.08);

            border-radius: 10px;

            overflow: hidden;
        }

        .progress-bar {
            height: 100%;

            background:
                linear-gradient(
                    90deg,
                    #7c3aed,
                    #c084fc,
                    #f0abfc
                );

            border-radius: 10px;

            transition: width 0.8s ease;
        }

        .model-info {
            margin-top: 25px;

            padding-top: 20px;

            border-top: 1px solid rgba(255,255,255,0.1);

            color: #777b91;

            font-size: 12px;

            text-align: center;

            line-height: 1.7;
        }

        .error {
            margin-top: 20px;

            padding: 15px;

            background: rgba(239,68,68,0.12);

            border: 1px solid rgba(239,68,68,0.3);

            border-radius: 12px;

            color: #fca5a5;

            font-size: 14px;
        }

        .footer {
            text-align: center;

            margin-top: 18px;

            color: #65687d;

            font-size: 12px;
        }

        @media (max-width: 600px) {

            body {
                padding: 15px;
            }

            .card {
                padding: 25px 20px;
            }

            h1 {
                font-size: 30px;
            }
        }

    </style>
</head>


<body>

<div class="container">

    <div class="card">

        <div class="badge">
            ML Powered Prediction
        </div>

        <h1>Placement Predictor</h1>

        <p class="subtitle">
            Enter your CGPA and Resume Score to predict
            your placement status using a Machine Learning model.
        </p>


        {% if error %}

            <div class="error">
                {{ error }}
            </div>

        {% endif %}


        <form method="POST">

            <div class="form-group">

                <label for="cgpa">
                    CGPA
                </label>

                <input
                    type="number"
                    id="cgpa"
                    name="cgpa"
                    step="0.01"
                    min="0"
                    max="10"
                    value="{{ cgpa }}"
                    placeholder="Enter CGPA"
                    required
                >

                <div class="hint">
                    Enter CGPA between 0 and 10
                </div>

            </div>


            <div class="form-group">

                <label for="resume_score">
                    Resume Score
                </label>

                <input
                    type="number"
                    id="resume_score"
                    name="resume_score"
                    step="0.01"
                    min="0"
                    max="10"
                    value="{{ resume_score }}"
                    placeholder="Enter Resume Score"
                    required
                >

                <div class="hint">
                    Enter Resume Score between 0 and 10
                </div>

            </div>


            <button
                type="submit"
                class="predict-btn"
            >
                🚀 Predict Status
            </button>

        </form>


        {% if prediction is not none %}

        <div class="result">

            <div class="result-title">
                Prediction Result
            </div>


            {% if prediction == 1 %}

                <div class="placed">
                    🎉 Placed
                </div>

                <div class="decision">
                    The model predicts that the candidate is placed.
                </div>

            {% else %}

                <div class="not-placed">
                    ❌ Not Placed
                </div>

                <div class="decision">
                    The model predicts that the candidate is not placed.
                </div>

            {% endif %}


            {% if decision_strength is not none %}

            <div class="confidence-box">

                <div class="confidence-header">

                    <span>
                        Decision Strength
                    </span>

                    <span>
                        {{ decision_strength }}%
                    </span>

                </div>

                <div class="progress">

                    <div
                        class="progress-bar"
                        style="width: {{ decision_strength }}%;"
                    ></div>

                </div>

            </div>

            {% endif %}


            <div class="model-info">

                <strong>Model:</strong> Perceptron<br>

                <strong>Features:</strong>
                CGPA + Resume Score<br>

                <strong>Prediction:</strong>
                Based on trained model

            </div>

        </div>

        {% endif %}


        <div class="footer">
            Built with Flask & Scikit-learn
        </div>

    </div>

</div>


{% if prediction == 1 %}

<script>

    // Balloons
    function createBalloons() {

        for (let i = 0; i < 25; i++) {

            const balloon = document.createElement("div");

            balloon.innerHTML = "🎈";

            balloon.style.position = "fixed";

            balloon.style.left =
                Math.random() * 100 + "%";

            balloon.style.bottom = "-50px";

            balloon.style.fontSize =
                (25 + Math.random() * 25) + "px";

            balloon.style.zIndex = "9999";

            balloon.style.pointerEvents = "none";

            balloon.style.animation =
                "floatBalloon " +
                (3 + Math.random() * 3) +
                "s linear forwards";

            document.body.appendChild(balloon);

            setTimeout(() => {
                balloon.remove();
            }, 6000);
        }
    }


    const style = document.createElement("style");

    style.innerHTML = `

        @keyframes floatBalloon {

            0% {
                transform:
                    translateY(0)
                    rotate(0deg);

                opacity: 1;
            }

            100% {
                transform:
                    translateY(-110vh)
                    rotate(360deg);

                opacity: 0;
            }

        }

    `;

    document.head.appendChild(style);

    createBalloons();

</script>

{% endif %}


</body>
</html>
"""


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------
def make_prediction(cgpa, resume_score):

    # Keep the exact feature order used during training
    input_data = np.array(
        [[cgpa, resume_score]],
        dtype=float
    )

    prediction = int(model.predict(input_data)[0])

    # Perceptron has decision_function but NOT predict_proba.
    # Therefore we call this "Decision Strength", not probability.
    try:

        decision_value = float(
            model.decision_function(input_data)[0]
        )

        # Convert decision margin to a visual 0-100 strength.
        # This is NOT a probability.
        decision_strength = (
            50 + (abs(decision_value) / (1 + abs(decision_value))) * 50
        )

        decision_strength = round(
            min(100, max(0, decision_strength)),
            1
        )

    except Exception:

        decision_strength = None

    return prediction, decision_strength


# ---------------------------------------------------------
# Home Route
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    decision_strength = None

    cgpa = ""
    resume_score = ""

    error = None

    if request.method == "POST":

        if not MODEL_LOADED:

            error = (
                "Model could not be loaded. "
                "Please check that perceptron.pkl is present."
            )

            return render_template_string(
                HTML,
                prediction=None,
                decision_strength=None,
                cgpa="",
                resume_score="",
                error=error
            )


        try:

            cgpa = float(request.form.get("cgpa", ""))

            resume_score = float(
                request.form.get("resume_score", "")
            )


            # Validation
            if not 0 <= cgpa <= 10:

                raise ValueError(
                    "CGPA must be between 0 and 10."
                )


            if not 0 <= resume_score <= 10:

                raise ValueError(
                    "Resume Score must be between 0 and 10."
                )


            prediction, decision_strength = make_prediction(
                cgpa,
                resume_score
            )


        except ValueError as e:

            error = str(e)

        except Exception as e:

            error = (
                "Prediction error: " + str(e)
            )


    return render_template_string(
        HTML,

        prediction=prediction,

        decision_strength=decision_strength,

        cgpa=cgpa,

        resume_score=resume_score,

        error=error
    )


# ---------------------------------------------------------
# Render / Production
# ---------------------------------------------------------
if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
