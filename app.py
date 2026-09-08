```python
from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------
# Load trained Perceptron model
# ---------------------------------------------------
with open("perceptron.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------
# HTML + CSS + JavaScript
# ---------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Perceptron ML Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left,
                    #4f46e5 0%,
                    transparent 30%),
                radial-gradient(circle at bottom right,
                    #9333ea 0%,
                    transparent 30%),
                #0f172a;

            color: #ffffff;
            padding: 30px;
        }

        /* ------------------------------------
           Main Dashboard
        ------------------------------------ */

        .dashboard {
            max-width: 1150px;
            margin: auto;
        }

        /* ------------------------------------
           Header
        ------------------------------------ */

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 35px;
            flex-wrap: wrap;
            gap: 20px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .brand-icon {
            width: 55px;
            height: 55px;
            border-radius: 16px;

            display: flex;
            justify-content: center;
            align-items: center;

            font-size: 27px;

            background:
                linear-gradient(135deg,
                #6366f1,
                #a855f7);

            box-shadow:
                0 10px 30px rgba(99,102,241,0.35);
        }

        .brand h1 {
            font-size: 26px;
        }

        .brand p {
            color: #94a3b8;
            font-size: 13px;
            margin-top: 3px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;

            padding: 9px 15px;

            border-radius: 30px;

            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.25);

            color: #86efac;

            font-size: 13px;
        }

        .status-dot {
            width: 8px;
            height: 8px;

            background: #22c55e;

            border-radius: 50%;

            box-shadow:
                0 0 12px #22c55e;
        }

        /* ------------------------------------
           Grid
        ------------------------------------ */

        .grid {
            display: grid;

            grid-template-columns:
                1.2fr 0.8fr;

            gap: 25px;
        }

        /* ------------------------------------
           Cards
        ------------------------------------ */

        .card {
            background: rgba(15,23,42,0.72);

            border: 1px solid
                rgba(255,255,255,0.09);

            border-radius: 22px;

            padding: 30px;

            backdrop-filter: blur(15px);

            box-shadow:
                0 20px 50px
                rgba(0,0,0,0.25);
        }

        .card-title {
            display: flex;
            align-items: center;
            gap: 10px;

            margin-bottom: 25px;
        }

        .card-title h2 {
            font-size: 19px;
        }

        .card-title span {
            font-size: 22px;
        }

        /* ------------------------------------
           Input
        ------------------------------------ */

        .input-group {
            margin-bottom: 23px;
        }

        .label-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 9px;
        }

        label {
            color: #e2e8f0;
            font-size: 14px;
            font-weight: 600;
        }

        .range {
            color: #818cf8;
            font-size: 12px;
        }

        input {
            width: 100%;

            padding: 15px 16px;

            border-radius: 13px;

            border: 1px solid
                rgba(255,255,255,0.12);

            background: rgba(255,255,255,0.055);

            color: white;

            font-size: 16px;

            outline: none;

            transition: 0.3s;
        }

        input::placeholder {
            color: #64748b;
        }

        input:focus {
            border-color: #818cf8;

            box-shadow:
                0 0 0 4px
                rgba(129,140,248,0.12);
        }

        /* ------------------------------------
           Predict Button
        ------------------------------------ */

        .predict-btn {
            width: 100%;

            padding: 16px;

            border: none;

            border-radius: 14px;

            color: white;

            font-size: 16px;

            font-weight: 700;

            cursor: pointer;

            background:
                linear-gradient(135deg,
                #6366f1,
                #a855f7);

            box-shadow:
                0 12px 30px
                rgba(99,102,241,0.28);

            transition: 0.3s;
        }

        .predict-btn:hover {
            transform: translateY(-3px);

            box-shadow:
                0 18px 35px
                rgba(99,102,241,0.4);
        }

        .predict-btn:active {
            transform: scale(0.98);
        }

        /* ------------------------------------
           Model Card
        ------------------------------------ */

        .model-icon {
            width: 70px;
            height: 70px;

            margin-bottom: 18px;

            border-radius: 20px;

            display: flex;
            justify-content: center;
            align-items: center;

            font-size: 34px;

            background:
                linear-gradient(135deg,
                rgba(99,102,241,0.25),
                rgba(168,85,247,0.25));

            border: 1px solid
                rgba(129,140,248,0.25);
        }

        .model-name {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .model-description {
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.6;
            margin-bottom: 25px;
        }

        .stats {
            display: grid;

            grid-template-columns: 1fr 1fr;

            gap: 12px;
        }

        .stat {
            padding: 16px;

            border-radius: 14px;

            background:
                rgba(255,255,255,0.04);

            border: 1px solid
                rgba(255,255,255,0.06);
        }

        .stat-label {
            color: #64748b;
            font-size: 11px;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 14px;
            font-weight: 600;
        }

        /* ------------------------------------
           Result
        ------------------------------------ */

        .result-card {
            margin-top: 25px;

            grid-column: 1 / -1;

            text-align: center;
        }

        .result-icon {
            width: 90px;
            height: 90px;

            margin: 5px auto 18px;

            border-radius: 50%;

            display: flex;
            justify-content: center;
            align-items: center;

            font-size: 42px;

            animation: pop 0.6s ease;
        }

        .success {
            background:
                rgba(34,197,94,0.12);

            border:
                1px solid rgba(34,197,94,0.3);

            box-shadow:
                0 0 35px rgba(34,197,94,0.18);
        }

        .danger {
            background:
                rgba(239,68,68,0.12);

            border:
                1px solid rgba(239,68,68,0.3);

            box-shadow:
                0 0 35px rgba(239,68,68,0.18);
        }

        .result-title {
            font-size: 28px;
            margin-bottom: 7px;
        }

        .result-subtitle {
            color: #94a3b8;
            font-size: 14px;
        }

        /* ------------------------------------
           Confidence
        ------------------------------------ */

        .confidence {
            max-width: 600px;
            margin: 28px auto 0;
            text-align: left;
        }

        .confidence-top {
            display: flex;
            justify-content: space-between;
            margin-bottom: 9px;
        }

        .confidence-top span:first-child {
            color: #cbd5e1;
            font-size: 13px;
        }

        .confidence-value {
            color: #818cf8;
            font-weight: 700;
        }

        .progress {
            height: 10px;

            border-radius: 20px;

            background: #1e293b;

            overflow: hidden;
        }

        .progress-bar {
            height: 100%;

            width: {{ confidence }}%;

            border-radius: 20px;

            background:
                linear-gradient(90deg,
                #6366f1,
                #a855f7);

            animation:
                progressAnimation 1.2s ease;
        }

        /* ------------------------------------
           Footer
        ------------------------------------ */

        .footer {
            text-align: center;

            margin-top: 30px;

            color: #64748b;

            font-size: 12px;
        }

        /* ------------------------------------
           Animations
        ------------------------------------ */

        @keyframes pop {

            0% {
                transform: scale(0);
                opacity: 0;
            }

            70% {
                transform: scale(1.15);
            }

            100% {
                transform: scale(1);
                opacity: 1;
            }
        }

        @keyframes progressAnimation {

            from {
                width: 0;
            }

            to {
                width: {{ confidence }}%;
            }
        }

        /* ------------------------------------
           Mobile
        ------------------------------------ */

        @media (max-width: 800px) {

            body {
                padding: 18px;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .result-card {
                grid-column: auto;
            }

            .header {
                align-items: flex-start;
            }

        }

    </style>

</head>


<body>

<div class="dashboard">

    <!-- HEADER -->

    <div class="header">

        <div class="brand">

            <div class="brand-icon">
                🤖
            </div>

            <div>

                <h1>Perceptron AI</h1>

                <p>
                    Machine Learning Prediction Dashboard
                </p>

            </div>

        </div>

        <div class="status">

            <div class="status-dot"></div>

            Model Online

        </div>

    </div>


    <!-- MAIN GRID -->

    <div class="grid">


        <!-- INPUT CARD -->

        <div class="card">

            <div class="card-title">

                <span>📊</span>

                <h2>Candidate Analysis</h2>

            </div>


            <form method="POST">


                <!-- CGPA -->

                <div class="input-group">

                    <div class="label-row">

                        <label>
                            Academic CGPA
                        </label>

                        <span class="range">
                            0 – 10
                        </span>

                    </div>

                    <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="10"
                        name="cgpa"
                        placeholder="Example: 8.5"
                        value="{{ cgpa }}"
                        required
                    >

                </div>


                <!-- Resume -->

                <div class="input-group">

                    <div class="label-row">

                        <label>
                            Resume Score
                        </label>

                        <span class="range">
                            0 – 100
                        </span>

                    </div>

                    <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="100"
                        name="resume_score"
                        placeholder="Example: 85"
                        value="{{ resume_score }}"
                        required
                    >

                </div>


                <button
                    class="predict-btn"
                    type="submit">

                    🚀 Run Prediction

                </button>

            </form>

        </div>


        <!-- MODEL INFORMATION -->

        <div class="card">

            <div class="card-title">

                <span>🧠</span>

                <h2>Model Information</h2>

            </div>


            <div class="model-icon">
                ⚡
            </div>

            <div class="model-name">
                Perceptron
            </div>

            <div class="model-description">

                A supervised machine learning
                classification algorithm used to
                predict binary outcomes based on
                input features.

            </div>


            <div class="stats">

                <div class="stat">

                    <div class="stat-label">
                        ALGORITHM
                    </div>

                    <div class="stat-value">
                        Perceptron
                    </div>

                </div>


                <div class="stat">

                    <div class="stat-label">
                        TASK
                    </div>

                    <div class="stat-value">
                        Classification
                    </div>

                </div>


                <div class="stat">

                    <div class="stat-label">
                        FEATURES
                    </div>

                    <div class="stat-value">
                        2 Inputs
                    </div>

                </div>


                <div class="stat">

                    <div class="stat-label">
                        OUTPUT
                    </div>

                    <div class="stat-value">
                        Binary
                    </div>

                </div>

            </div>

        </div>


        {% if prediction is not none %}

        <!-- RESULT -->

        <div class="card result-card">

            {% if prediction == 1 %}

                <div class="result-icon success">
                    🎯
                </div>

                <div class="result-title">
                    Candidate Selected
                </div>

                <div class="result-subtitle">
                    The model predicts a positive outcome.
                </div>

            {% else %}

                <div class="result-icon danger">
                    ⚠️
                </div>

                <div class="result-title">
                    Candidate Not Selected
                </div>

                <div class="result-subtitle">
                    The model predicts a negative outcome.
                </div>

            {% endif %}


            <!-- CONFIDENCE -->

            <div class="confidence">

                <div class="confidence-top">

                    <span>
                        Model Confidence
                    </span>

                    <span class="confidence-value">
                        {{ confidence }}%
                    </span>

                </div>

                <div class="progress">

                    <div class="progress-bar"></div>

                </div>

            </div>

        </div>

        {% endif %}


        {% if error %}

        <div class="card result-card">

            <div class="result-icon danger">
                ❌
            </div>

            <div class="result-title">
                Invalid Input
            </div>

            <div class="result-subtitle">
                {{ error }}
            </div>

        </div>

        {% endif %}

    </div>


    <div class="footer">

        Built with Python • Flask • Scikit-learn • Perceptron

    </div>

</div>


</body>

</html>
"""


# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = 0
    error = None

    cgpa = ""
    resume_score = ""

    if request.method == "POST":

        try:

            cgpa = float(request.form["cgpa"])
            resume_score = float(request.form["resume_score"])

            # Validation

            if cgpa < 0 or cgpa > 10:
                raise ValueError(
                    "CGPA must be between 0 and 10."
                )

            if resume_score < 0 or resume_score > 100:
                raise ValueError(
                    "Resume Score must be between 0 and 100."
                )

            # Model input
            input_data = np.array([
                [cgpa, resume_score]
            ])

            # Prediction
            prediction = int(
                model.predict(input_data)[0]
            )

            # -----------------------------------------
            # Confidence
            # -----------------------------------------
            #
            # Perceptron normally does not provide
            # calibrated probabilities.
            #
            # We therefore use its decision margin
            # to create a confidence-style score.
            # -----------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                confidence = float(
                    max(probabilities) * 100
                )

            elif hasattr(model, "decision_function"):

                decision = float(
                    model.decision_function(input_data)[0]
                )

                # Convert decision margin into
                # confidence-style percentage.

                confidence = (
                    1 /
                    (1 + np.exp(-abs(decision)))
                ) * 100

            else:

                confidence = 100


            confidence = round(
                min(max(confidence, 0), 100),
                1
            )

        except ValueError as e:

            error = str(e)

        except Exception:

            error = (
                "Unable to make prediction. "
                "Please check the model and input values."
            )


    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        error=error,
        cgpa=cgpa,
        resume_score=resume_score
    )


# ---------------------------------------------------
# Run Application
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
```
