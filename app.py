from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Keep perceptron(2).pkl in the same folder as app.py
MODEL_PATH = "perceptron(2).pkl"
model = joblib.load(MODEL_PATH)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Career Placement Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            min-height: 100vh;
            font-family: Inter, Arial, sans-serif;
            background:
                radial-gradient(circle at 10% 10%, rgba(99,102,241,.35), transparent 28%),
                radial-gradient(circle at 90% 20%, rgba(236,72,153,.28), transparent 28%),
                linear-gradient(135deg, #090d1a, #111936 55%, #160d25);
            color: #fff;
            overflow-x: hidden;
        }
        .wrap { max-width: 1050px; margin: auto; padding: 35px 20px 50px; }
        .hero { text-align: center; margin: 20px auto 30px; }
        .badge {
            display: inline-block; padding: 8px 15px; border-radius: 999px;
            background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16);
            font-size: 13px; letter-spacing: .5px; color: #c7d2fe;
        }
        h1 { font-size: clamp(34px, 6vw, 64px); margin: 15px 0 8px; line-height: 1.05; }
        .gradient { background: linear-gradient(90deg,#a5b4fc,#f9a8d4,#fde68a);
            -webkit-background-clip:text; background-clip:text; color:transparent; }
        .hero p { color: #b8c1d9; font-size: 16px; }
        .card {
            max-width: 620px; margin: auto; padding: 32px;
            background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
            border-radius: 28px; backdrop-filter: blur(18px);
            box-shadow: 0 25px 70px rgba(0,0,0,.35);
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .field { margin-bottom: 4px; }
        label { display:block; margin-bottom:8px; font-weight:700; color:#e8ecf8; }
        .hint { color:#8995b3; font-size:12px; margin-top:6px; }
        input {
            width:100%; padding:15px 16px; border-radius:14px;
            border:1px solid rgba(255,255,255,.15); outline:none;
            background:rgba(7,11,25,.65); color:white; font-size:16px;
            transition:.2s;
        }
        input:focus { border-color:#a78bfa; box-shadow:0 0 0 4px rgba(167,139,250,.12); }
        button {
            width:100%; margin-top:23px; padding:16px; border:0; border-radius:15px;
            cursor:pointer; color:#fff; font-size:17px; font-weight:800;
            background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);
            box-shadow:0 12px 30px rgba(139,92,246,.35); transition:.2s;
        }
        button:hover { transform:translateY(-2px); box-shadow:0 16px 35px rgba(236,72,153,.35); }
        .result {
            margin-top:22px; padding:22px; border-radius:20px; text-align:center;
            background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
        }
        .result h2 { font-size:30px; margin:7px 0; }
        .success { color:#86efac; }
        .danger { color:#fda4af; }
        .small { color:#aeb8d0; font-size:13px; margin-top:7px; }
        .footer { text-align:center; color:#697590; margin-top:25px; font-size:13px; }
        .balloon {
            position:fixed; bottom:-120px; width:42px; height:52px; border-radius:50% 50% 48% 48%;
            z-index:50; animation:floatUp linear forwards; pointer-events:none;
        }
        .balloon:after {
            content:""; position:absolute; left:19px; bottom:-15px; width:4px; height:18px;
            background:rgba(255,255,255,.65);
        }
        @keyframes floatUp {
            0% { transform:translateY(0) rotate(0deg); opacity:1; }
            100% { transform:translateY(-115vh) rotate(20deg); opacity:.85; }
        }
        @media(max-width:650px) {
            .grid { grid-template-columns:1fr; }
            .card { padding:23px; }
        }
    </style>
</head>
<body>
<div class="wrap">
    <section class="hero">
        <span class="badge">⚡ AI • PERCEPTRON MODEL</span>
        <h1><span class="gradient">Career Placement</span><br>Predictor</h1>
        <p>Enter your academic and resume scores to check the predicted placement status.</p>
    </section>

    <div class="card">
        <form method="POST" onsubmit="celebrate(event)">
            <div class="grid">
                <div class="field">
                    <label for="cgpa">🎓 CGPA</label>
                    <input id="cgpa" name="cgpa" type="number" step="0.01" min="0" max="10"
                           placeholder="e.g. 7.50" value="{{ cgpa }}" required>
                    <div class="hint">Enter CGPA between 0 and 10</div>
                </div>

                <div class="field">
                    <label for="resume_score">📄 Resume Score</label>
                    <input id="resume_score" name="resume_score" type="number" step="0.01" min="0"
                           max="100" placeholder="e.g. 75" value="{{ resume_score }}" required>
                    <div class="hint">Enter resume score between 0 and 100</div>
                </div>
            </div>

            <button type="submit">🚀 Predict Placement Status</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            {% if prediction == 1 %}
                <div style="font-size:40px">🎉</div>
                <h2 class="success">Placed / Eligible</h2>
                <div class="small">The Perceptron model predicted a positive placement result.</div>
            {% else %}
                <div style="font-size:40px">💪</div>
                <h2 class="danger">Not Placed / Not Eligible</h2>
                <div class="small">The Perceptron model predicted a negative placement result.</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
    <div class="footer">Built with Flask + Scikit-learn Perceptron • Ready for Render</div>
</div>

<script>
function celebrate(e) {
    // Let Flask process the form, while starting the balloon animation immediately.
    for (let i = 0; i < 28; i++) {
        const b = document.createElement("div");
        b.className = "balloon";
        b.style.left = (Math.random() * 100) + "vw";
        b.style.animationDuration = (3 + Math.random() * 3) + "s";
        b.style.animationDelay = (Math.random() * .8) + "s";
        b.style.transform = "rotate(" + ((Math.random()*30)-15) + "deg)";
        const colors = ["#ff4d8d","#8b5cf6","#22d3ee","#facc15","#34d399","#fb7185"];
        b.style.background = colors[Math.floor(Math.random()*colors.length)];
        document.body.appendChild(b);
        setTimeout(() => b.remove(), 7000);
    }
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    cgpa = ""
    resume_score = ""

    if request.method == "POST":
        try:
            cgpa = float(request.form["cgpa"])
            resume_score = float(request.form["resume_score"])

            if not 0 <= cgpa <= 10:
                raise ValueError("CGPA must be between 0 and 10.")
            if not 0 <= resume_score <= 100:
                raise ValueError("Resume score must be between 0 and 100.")

            # IMPORTANT: feature order must match the model's training order.
            X = np.array([[cgpa, resume_score]], dtype=float)
            prediction = int(model.predict(X)[0])

        except (ValueError, TypeError, KeyError):
            prediction = None

    return render_template_string(
        HTML,
        prediction=prediction,
        cgpa=cgpa,
        resume_score=resume_score
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
