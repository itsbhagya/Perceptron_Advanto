import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved Perceptron model
MODEL_PATH = "perceptron.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    status = None
    
    if request.method == "POST":
        try:
            cgpa = float(request.form.get("cgpa"))
            resume_score = float(request.form.get("resume_score"))
            
            if model:
                # Features: [cgpa, resume_score]
                input_data = np.array([[cgpa, resume_score]])
                prediction = model.predict(input_data)[0]
                
                if prediction == 1:
                    prediction_text = "Congratulations! High Placement Chance."
                    status = "success"
                else:
                    prediction_text = "Needs Improvement. Low Placement Chance."
                    status = "danger"
            else:
                prediction_text = "Model file not found!"
                status = "error"
        except Exception as e:
            prediction_text = f"Error processing input: {str(e)}"
            status = "error"

    return render_template("index.html", prediction_text=prediction_text, status=status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placement Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-slate-800 border border-slate-700 shadow-2xl rounded-2xl p-8 max-w-md w-full">
        <h1 class="text-3xl font-extrabold text-center text-indigo-400 mb-2">Placement Predictor</h1>
        <p class="text-slate-400 text-sm text-center mb-6">Perceptron Machine Learning Model</p>

        <form method="POST" class="space-y-5">
            <div>
                <label for="cgpa" class="block text-sm font-medium text-slate-300 mb-1">CGPA (e.g., 8.5)</label>
                <input type="number" step="0.01" min="0" max="10" name="cgpa" id="cgpa" required 
                       class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none text-white">
            </div>

            <div>
                <label for="resume_score" class="block text-sm font-medium text-slate-300 mb-1">Resume Score (e.g., 7.5)</label>
                <input type="number" step="0.01" min="0" max="10" name="resume_score" id="resume_score" required 
                       class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none text-white">
            </div>

            <button type="submit" 
                    class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-lg transition duration-200 shadow-lg hover:shadow-indigo-500/25">
                Predict Outcome
            </button>
        </form>

        {% if prediction_text %}
        <div class="mt-6 p-4 rounded-lg text-center font-medium
            {% if status == 'success' %} bg-emerald-950 border border-emerald-700 text-emerald-300
            {% elif status == 'danger' %} bg-rose-950 border border-rose-700 text-rose-300
            {% else %} bg-amber-950 border border-amber-700 text-amber-300 {% endif %}">
            {{ prediction_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
