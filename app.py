from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Perceptron model
with open("perceptron.pkl", "rb") as file:
    model = pickle.load(file)


HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Placement Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {

            min-height: 100vh;

            display: flex;

            justify-content: center;

            align-items: center;

            padding: 20px;

            font-family:
                "Segoe UI",
                Arial,
                sans-serif;

            background:
                radial-gradient(
                    circle at 20% 20%,
                    rgba(92, 82, 200, 0.20),
                    transparent 35%
                ),
                radial-gradient(
                    circle at 80% 80%,
                    rgba(120, 60, 200, 0.15),
                    transparent 35%
                ),
                linear-gradient(
                    135deg,
                    #101331,
                    #171a42,
                    #202052
                );

            color: white;
        }


        /* Main Card */

        .card {

            width: 100%;

            max-width: 550px;

            padding: 46px 36px;

            border-radius: 28px;

            background:
                rgba(37, 38, 82, 0.82);

            border:
                1px solid
                rgba(255, 255, 255, 0.10);

            box-shadow:
                0 30px 80px
                rgba(0, 0, 0, 0.35);

            backdrop-filter: blur(18px);

            animation:
                cardAppear 0.7s ease;
        }


        /* Heading */

        .header {

            text-align: center;

            margin-bottom: 35px;
        }


        .header h1 {

            font-size: 38px;

            font-weight: 700;

            letter-spacing: -1px;

            color: #f4f4ff;

            margin-bottom: 8px;
        }


        .header p {

            font-size: 16px;

            color: #a5a7c5;

        }


        /* Form */

        .form-group {

            margin-bottom: 25px;
        }


        .form-group label {

            display: block;

            margin-bottom: 10px;

            font-size: 16px;

            font-weight: 600;

            color: #aeb0ca;
        }


        .input-wrapper {

            position: relative;
        }


        input {

            width: 100%;

            height: 62px;

            padding:
                0 20px;

            border-radius: 15px;

            border:
                1px solid
                rgba(255, 255, 255, 0.10);

            outline: none;

            background:
                rgba(12, 16, 43, 0.72);

            color: #ffffff;

            font-size: 18px;

            transition: all 0.3s ease;
        }


        input::placeholder {

            color: #626581;
        }


        input:focus {

            border-color: #7568ff;

            box-shadow:
                0 0 0 4px
                rgba(117, 104, 255, 0.12);

            background:
                rgba(12, 16, 43, 0.9);
        }


        /* Predict Button */

        .predict-btn {

            width: 100%;

            height: 60px;

            margin-top: 8px;

            border: none;

            border-radius: 15px;

            background:
                linear-gradient(
                    90deg,
                    #6266f5,
                    #a64df0
                );

            color: white;

            font-size: 18px;

            font-weight: 700;

            cursor: pointer;

            box-shadow:
                0 12px 30px
                rgba(120, 80, 240, 0.30);

            transition:
                transform 0.25s ease,
                box-shadow 0.25s ease;
        }


        .predict-btn:hover {

            transform:
                translateY(-2px);

            box-shadow:
                0 16px 35px
                rgba(120, 80, 240, 0.42);
        }


        .predict-btn:active {

            transform:
                scale(0.98);
        }


        /* Result */

        .result {

            margin-top: 35px;

            padding: 20px;

            border-radius: 15px;

            text-align: center;

            animation:
                resultAppear 0.55s ease;
        }


        .result.placed {

            background:
                rgba(34, 197, 94, 0.12);

            border:
                1px solid
                rgba(34, 197, 94, 0.35);
        }


        .result.not-placed {

            background:
                rgba(239, 68, 68, 0.12);

            border:
                1px solid
                rgba(239, 68, 68, 0.35);
        }


        .result-title {

            font-size: 20px;

            font-weight: 700;

            margin-bottom: 12px;
        }


        .placed .result-title {

            color: #69e89b;
        }


        .not-placed .result-title {

            color: #ff7777;
        }


        /* Confidence */

        .confidence {

            margin-top: 17px;

            text-align: left;
        }


        .confidence-header {

            display: flex;

            justify-content: space-between;

            margin-bottom: 8px;

            color: #aeb0ca;

            font-size: 13px;
        }


        .confidence-value {

            color: #b8aaff;

            font-weight: 700;
        }


        .progress {

            width: 100%;

            height: 9px;

            background:
                rgba(255, 255, 255, 0.08);

            border-radius: 20px;

            overflow: hidden;
        }


        .progress-bar {

            height: 100%;

            width: {{ confidence }}%;

            border-radius: 20px;

            background:
                linear-gradient(
                    90deg,
                    #6366f1,
                    #b04cf3
                );

            animation:
                progress 1.2s ease;
        }


        /* Model Info */

        .model-info {

            text-align: center;

            margin-top: 25px;

            color: #777b9e;

            font-size: 12px;

            line-height: 1.6;
        }


        .model-info strong {

            color: #999cc0;
        }


        /* Error */

        .error {

            margin-top: 25px;

            padding: 15px;

            border-radius: 13px;

            text-align: center;

            color: #ff8585;

            background:
                rgba(239, 68, 68, 0.10);

            border:
                1px solid
                rgba(239, 68, 68, 0.25);

            animation:
                resultAppear 0.5s ease;
        }


        /* Animations */

        @keyframes cardAppear {

            from {

                opacity: 0;

                transform:
                    translateY(25px)
                    scale(0.97);
            }

            to {

                opacity: 1;

                transform:
                    translateY(0)
                    scale(1);
            }
        }


        @keyframes resultAppear {

            from {

                opacity: 0;

                transform:
                    translateY(15px);
            }

            to {

                opacity: 1;

                transform:
                    translateY(0);
            }
        }


        @keyframes progress {

            from {

                width: 0%;
            }

            to {

                width: {{ confidence }}%;
            }
        }


        /* Mobile */

        @media (max-width: 600px) {

            body {

                padding: 15px;
            }

            .card {

                padding:
                    35px 22px;

                border-radius: 22px;
            }

            .header h1 {

                font-size: 30px;
            }

            .header p {

                font-size: 14px;
            }

            input {

                height: 58px;
            }

            .predict-btn {

                height: 58px;
            }
        }

    </style>

</head>


<body>


<div class="card">


    <!-- Header -->

    <div class="header">

        <h1>
            Placement Predictor
        </h1>

        <p>
            Perceptron Categorical Classifier
        </p>

    </div>


    <!-- Form -->

    <form method="POST">


        <!-- CGPA -->

        <div class="form-group">

            <label>
                CGPA (0 – 10)
            </label>

            <div class="input-wrapper">

                <input
                    type="number"
                    name="cgpa"
                    step="0.1"
                    min="0"
                    max="10"
                    placeholder="Enter your CGPA"
                    value="{{ cgpa }}"
                    required
                >

            </div>

        </div>


        <!-- Resume Score -->

        <div class="form-group">

            <label>
                Resume Score (0 – 10)
            </label>

            <div class="input-wrapper">

                <input
                    type="number"
                    name="resume_score"
                    step="0.1"
                    min="0"
                    max="10"
                    placeholder="Enter your resume score"
                    value="{{ resume_score }}"
                    required
                >

            </div>

        </div>


        <!-- Button -->

        <button
            type="submit"
            class="predict-btn">

            Predict Status

        </button>


    </form>


    <!-- Result -->

    {% if prediction is not none %}

        {% if prediction == 1 %}

            <div class="result placed">

                <div class="result-title">

                    🎉 Category: Placed

                </div>

                <div class="confidence">

                    <div class="confidence-header">

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

        {% else %}

            <div class="result not-placed">

                <div class="result-title">

                    ⚠️ Category: Not Placed

                </div>

                <div class="confidence">

                    <div class="confidence-header">

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

    {% endif %}


    <!-- Error -->

    {% if error %}

        <div class="error">

            ⚠️ {{ error }}

        </div>

    {% endif %}


    <!-- Model Info -->

    <div class="model-info">

        <strong>Model:</strong> Perceptron<br>

        <strong>Features:</strong>
        CGPA + Resume Score<br>

        <strong>Task:</strong>
        Binary Classification

    </div>


</div>


</body>

</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    confidence = 0

    cgpa = ""

    resume_score = ""

    error = None


    if request.method == "POST":

        try:

            # Get input values

            cgpa = float(
                request.form["cgpa"]
            )

            resume_score = float(
                request.form["resume_score"]
            )


            # Validate CGPA

            if cgpa < 0 or cgpa > 10:

                raise ValueError(
                    "CGPA must be between 0 and 10."
                )


            # Validate Resume Score

            if resume_score < 0 or resume_score > 10:

                raise ValueError(
                    "Resume Score must be between 0 and 10."
                )


            # Prepare input

            input_data = np.array(
                [[cgpa, resume_score]]
            )


            # Prediction

            prediction = int(
                model.predict(input_data)[0]
            )


            # ---------------------------------
            # Confidence
            # ---------------------------------

            if hasattr(
                model,
                "decision_function"
            ):

                decision = float(
                    model.decision_function(
                        input_data
                    )[0]
                )

                confidence = (
                    1 /
                    (
                        1 +
                        np.exp(-abs(decision))
                    )
                ) * 100

            else:

                confidence = 100.0


            confidence = round(
                min(
                    max(confidence, 0),
                    100
                ),
                1
            )


        except ValueError as e:

            error = str(e)


        except Exception as e:

            error = (
                "Prediction failed. "
                "Please check your model."
            )


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        cgpa=cgpa,

        resume_score=resume_score,

        error=error

    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
