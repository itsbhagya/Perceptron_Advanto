from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Perceptron model
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
        Arial,
        Helvetica,
        sans-serif;

    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(89, 80, 190, 0.25),
            transparent 35%
        ),
        radial-gradient(
            circle at 80% 80%,
            rgba(132, 70, 210, 0.20),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #10132f,
            #171942,
            #202052
        );

    color: white;

    overflow-x: hidden;
}


/* Main Card */

.card {

    width: 100%;

    max-width: 550px;

    padding: 42px 36px;

    border-radius: 27px;

    background:
        rgba(37, 38, 82, 0.88);

    border:
        1px solid
        rgba(255,255,255,0.10);

    box-shadow:
        0 30px 80px
        rgba(0,0,0,0.40);

    backdrop-filter:
        blur(18px);

    animation:
        cardAnimation 0.7s ease;

    position: relative;

    z-index: 2;
}


/* Heading */

.header {

    text-align: center;

    margin-bottom: 34px;
}


.header h1 {

    font-size: 36px;

    font-weight: 700;

    color: #f5f5ff;

    margin-bottom: 8px;
}


.header p {

    font-size: 15px;

    color: #a3a5c4;
}


/* Input */

.form-group {

    margin-bottom: 24px;
}


.form-group label {

    display: block;

    margin-bottom: 10px;

    color: #afb1ca;

    font-size: 15px;

    font-weight: 600;
}


input {

    width: 100%;

    height: 61px;

    padding:
        0 18px;

    border-radius: 14px;

    border:
        1px solid
        rgba(255,255,255,0.10);

    background:
        rgba(10,15,40,0.78);

    color: white;

    font-size: 17px;

    outline: none;

    transition: 0.3s;
}


input::placeholder {

    color: #60647f;
}


input:focus {

    border-color: #7669ff;

    box-shadow:
        0 0 0 4px
        rgba(118,105,255,0.12);
}


/* Button */

.predict-btn {

    width: 100%;

    height: 60px;

    margin-top: 8px;

    border: none;

    border-radius: 14px;

    background:
        linear-gradient(
            90deg,
            #6266f5,
            #a64df0
        );

    color: white;

    font-size: 17px;

    font-weight: 700;

    cursor: pointer;

    box-shadow:
        0 12px 30px
        rgba(116,83,240,0.35);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.predict-btn:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 16px 35px
        rgba(116,83,240,0.50);
}


.predict-btn:active {

    transform:
        scale(0.97);
}


/* Result */

.result {

    margin-top: 32px;

    padding: 19px;

    border-radius: 14px;

    text-align: center;

    animation:
        resultAnimation 0.55s ease;
}


.result.placed {

    background:
        rgba(34,197,94,0.12);

    border:
        1px solid
        rgba(34,197,94,0.30);
}


.result.not-placed {

    background:
        rgba(239,68,68,0.12);

    border:
        1px solid
        rgba(239,68,68,0.30);
}


.result-title {

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 15px;
}


.placed .result-title {

    color: #67e89a;
}


.not-placed .result-title {

    color: #ff7474;
}


/* Confidence */

.confidence {

    margin-top: 10px;

    text-align: left;
}


.confidence-header {

    display: flex;

    justify-content: space-between;

    margin-bottom: 8px;

    font-size: 12px;

    color: #aeb0c7;
}


.confidence-value {

    color: #b5a7ff;

    font-weight: bold;
}


.progress {

    height: 8px;

    width: 100%;

    background:
        rgba(255,255,255,0.08);

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
            #6266f5,
            #b04cf3
        );

    animation:
        progressAnimation 1s ease;
}


/* Model Information */

.model-info {

    margin-top: 25px;

    text-align: center;

    color: #74789b;

    font-size: 11px;

    line-height: 1.7;
}


.model-info strong {

    color: #9699b8;
}


/* Balloons */

.balloon {

    position: fixed;

    bottom: -100px;

    width: 38px;

    height: 48px;

    border-radius:
        50% 50% 45% 45%;

    z-index: 10;

    animation:
        balloonUp linear forwards;

    pointer-events: none;
}


.balloon::after {

    content: "";

    position: absolute;

    bottom: -22px;

    left: 50%;

    width: 1px;

    height: 25px;

    background:
        rgba(255,255,255,0.55);
}


.balloon::before {

    content: "";

    position: absolute;

    bottom: -5px;

    left: 50%;

    transform:
        translateX(-50%);

    width: 0;

    height: 0;

    border-left:
        6px solid transparent;

    border-right:
        6px solid transparent;

    border-top:
        9px solid currentColor;
}


@keyframes balloonUp {

    0% {

        transform:
            translateY(0)
            rotate(0deg);

        opacity: 1;
    }

    25% {

        transform:
            translateY(-25vh)
            rotate(8deg);
    }

    50% {

        transform:
            translateY(-50vh)
            rotate(-8deg);
    }

    75% {

        transform:
            translateY(-75vh)
            rotate(8deg);
    }

    100% {

        transform:
            translateY(-120vh)
            rotate(-8deg);

        opacity: 0;
    }
}


/* Confetti */

.confetti {

    position: fixed;

    top: -20px;

    width: 9px;

    height: 15px;

    z-index: 11;

    animation:
        confettiFall linear forwards;

    pointer-events: none;
}


@keyframes confettiFall {

    0% {

        transform:
            translateY(0)
            rotate(0deg);

        opacity: 1;
    }

    100% {

        transform:
            translateY(110vh)
            rotate(720deg);

        opacity: 0;
    }
}


/* Animations */

@keyframes cardAnimation {

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


@keyframes resultAnimation {

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


@keyframes progressAnimation {

    from {

        width: 0%;
    }

    to {

        width: {{ confidence }}%;
    }
}


/* Mobile */

@media(max-width:600px) {

    .card {

        padding:
            35px 22px;
    }


    .header h1 {

        font-size: 30px;
    }


    input {

        height: 57px;
    }


    .predict-btn {

        height: 57px;
    }

}

</style>

</head>


<body>


<div class="card">


    <div class="header">

        <h1>
            Placement Predictor
        </h1>

        <p>
            Perceptron Categorical Classifier
        </p>

    </div>


    <form method="POST">


        <div class="form-group">

            <label>
                CGPA (0 – 10)
            </label>

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


        <div class="form-group">

            <label>
                Resume Score (0 – 10)
            </label>

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


        <button
            type="submit"
            class="predict-btn">

            Predict Status

        </button>


    </form>


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


            <script>

                window.onload = function() {

                    createBalloons();

                    createConfetti();

                };


                function createBalloons() {

                    const colors = [

                        "#ff4d6d",
                        "#ffd166",
                        "#06d6a0",
                        "#4dabf7",
                        "#c77dff",
                        "#ff9f1c"

                    ];


                    for (
                        let i = 0;
                        i < 22;
                        i++
                    ) {

                        const balloon =
                            document.createElement(
                                "div"
                            );

                        balloon.className =
                            "balloon";


                        balloon.style.left =
                            Math.random() * 100 + "%";


                        balloon.style.background =
                            colors[
                                Math.floor(
                                    Math.random() *
                                    colors.length
                                )
                            ];


                        balloon.style.color =
                            balloon.style.background;


                        balloon.style.animationDuration =
                            (4 + Math.random() * 4)
                            + "s";


                        balloon.style.animationDelay =
                            (Math.random() * 1.5)
                            + "s";


                        document.body.appendChild(
                            balloon
                        );


                        setTimeout(
                            function() {

                                balloon.remove();

                            },
                            9000
                        );

                    }

                }


                function createConfetti() {

                    const colors = [

                        "#6366f1",
                        "#a855f7",
                        "#facc15",
                        "#22c55e",
                        "#f43f5e",
                        "#38bdf8"

                    ];


                    for (
                        let i = 0;
                        i < 70;
                        i++
                    ) {

                        const piece =
                            document.createElement(
                                "div"
                            );

                        piece.className =
                            "confetti";


                        piece.style.left =
                            Math.random() * 100 + "%";


                        piece.style.background =
                            colors[
                                Math.floor(
                                    Math.random() *
                                    colors.length
                                )
                            ];


                        piece.style.animationDuration =
                            (2 + Math.random() * 3)
                            + "s";


                        piece.style.animationDelay =
                            (Math.random() * 1)
                            + "s";


                        document.body.appendChild(
                            piece
                        );


                        setTimeout(
                            function() {

                                piece.remove();

                            },
                            6000
                        );

                    }

                }

            </script>


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


    {% if error %}

        <div class="error">

            ⚠️ {{ error }}

        </div>

    {% endif %}


    <div class="model-info">

        <strong>Model:</strong>
        Perceptron

        &nbsp; | &nbsp;

        <strong>Features:</strong>
        CGPA + Resume Score

        <br>

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

            # Get values
            cgpa = float(
                request.form["cgpa"]
            )

            resume_score = float(
                request.form["resume_score"]
            )


            # Validation

            if cgpa < 0 or cgpa > 10:

                raise ValueError(
                    "CGPA must be between 0 and 10."
                )


            if resume_score < 0 or resume_score > 10:

                raise ValueError(
                    "Resume Score must be between 0 and 10."
                )


            # IMPORTANT:
            # Use DataFrame-like input with
            # correct feature names.

            try:

                import pandas as pd

                input_data = pd.DataFrame(
                    {
                        "cgpa": [cgpa],
                        "resume_score": [resume_score]
                    }
                )

            except ImportError:

                input_data = np.array(
                    [[cgpa, resume_score]]
                )


            # Prediction

            prediction = int(
                model.predict(input_data)[0]
            )


            # Confidence

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
                        np.exp(
                            -abs(decision)
                        )
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
                "Please check the model and inputs."
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
