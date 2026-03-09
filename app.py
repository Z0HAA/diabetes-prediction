"""
Diabetes Prediction Web Application
=====================================
Flask web app for PythonAnywhere deployment.
Uses MLP Neural Network + StandardScaler on all 8 features.
"""

from flask import Flask, request, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load model AND scaler at startup — both are required
model  = pickle.load(open('mlp_trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex;
            align-items: center; justify-content: center; padding: 20px;
        }
        .container {
            background: white; border-radius: 16px; padding: 40px;
            width: 100%; max-width: 540px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { text-align:center; color:#4a3f8f; font-size:1.6rem; margin-bottom:6px; }
        .subtitle { text-align:center; color:#888; font-size:0.85rem; margin-bottom:26px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px; }
        label { display:block; font-weight:600; color:#444; margin-bottom:5px; margin-top:16px; font-size:0.88rem; }
        input[type=number] {
            width:100%; padding:10px 12px; border:2px solid #e0e0e0;
            border-radius:8px; font-size:0.92rem; transition:border-color 0.2s;
        }
        input[type=number]:focus { outline:none; border-color:#667eea; }
        .hint { color:#bbb; font-size:0.73rem; margin-top:3px; }
        button {
            width:100%; padding:13px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color:white; border:none; border-radius:8px; font-size:1.05rem;
            font-weight:700; cursor:pointer; margin-top:26px; transition:opacity 0.2s;
        }
        button:hover { opacity:0.9; }
        .result {
            margin-top:22px; padding:16px; border-radius:10px;
            text-align:center; font-size:1.2rem; font-weight:700;
        }
        .diabetic     { background:#fdecea; color:#c62828; border:2px solid #ef9a9a; }
        .not-diabetic { background:#e8f5e9; color:#2e7d32; border:2px solid #a5d6a7; }
        .footer { text-align:center; margin-top:18px; color:#bbb; font-size:0.73rem; }
    </style>
</head>
<body>
<div class="container">
    <h1>Diabetes Prediction</h1>
    <p class="subtitle">MLP Neural Network &mdash; Pima Indians Diabetes Database</p>

    <form method="POST" action="/predict">
        <div class="grid">
            <div>
                <label>Pregnancies</label>
                <input type="number" name="pregnancies" min="0" max="20" step="1"
                       value="{{ vals.pregnancies or '' }}" placeholder="e.g. 2" required>
                <p class="hint">Number of pregnancies</p>
            </div>
            <div>
                <label>Glucose (mg/dL)</label>
                <input type="number" name="glucose" min="0" max="300" step="1"
                       value="{{ vals.glucose or '' }}" placeholder="e.g. 120" required>
                <p class="hint">Plasma glucose level</p>
            </div>
            <div>
                <label>Blood Pressure (mm Hg)</label>
                <input type="number" name="bp" min="0" max="200" step="1"
                       value="{{ vals.bp or '' }}" placeholder="e.g. 72" required>
                <p class="hint">Diastolic blood pressure</p>
            </div>
            <div>
                <label>Skin Thickness (mm)</label>
                <input type="number" name="skin" min="0" max="100" step="1"
                       value="{{ vals.skin or '' }}" placeholder="e.g. 20" required>
                <p class="hint">Triceps skin fold</p>
            </div>
            <div>
                <label>Insulin (mu U/ml)</label>
                <input type="number" name="insulin" min="0" max="900" step="1"
                       value="{{ vals.insulin or '' }}" placeholder="e.g. 80" required>
                <p class="hint">2-Hour serum insulin</p>
            </div>
            <div>
                <label>BMI (kg/m²)</label>
                <input type="number" name="bmi" min="0" max="80" step="0.1"
                       value="{{ vals.bmi or '' }}" placeholder="e.g. 28.5" required>
                <p class="hint">Body Mass Index</p>
            </div>
            <div>
                <label>Diabetes Pedigree</label>
                <input type="number" name="dpf" min="0" max="3" step="0.001"
                       value="{{ vals.dpf or '' }}" placeholder="e.g. 0.627" required>
                <p class="hint">Pedigree function score</p>
            </div>
            <div>
                <label>Age (years)</label>
                <input type="number" name="age" min="21" max="100" step="1"
                       value="{{ vals.age or '' }}" placeholder="e.g. 35" required>
                <p class="hint">Minimum age: 21</p>
            </div>
        </div>

        <button type="submit">Predict Diabetes</button>
    </form>

    {% if prediction %}
    <div class="result {{ 'diabetic' if prediction == 'DIABETIC' else 'not-diabetic' }}">
        {% if prediction == 'DIABETIC' %}
            Prediction: DIABETIC
        {% else %}
            Prediction: NOT DIABETIC
        {% endif %}
    </div>
    {% endif %}

    <p class="footer">For educational purposes only. Always consult a medical professional for diagnosis.</p>
</div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None, vals={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vals = {k: request.form[k] for k in ['pregnancies','glucose','bp','skin','insulin','bmi','dpf','age']}
        user_input = pd.DataFrame({
            'Pregnancies':              [float(vals['pregnancies'])],
            'Glucose':                  [float(vals['glucose'])],
            'BloodPressure':            [float(vals['bp'])],
            'SkinThickness':            [float(vals['skin'])],
            'Insulin':                  [float(vals['insulin'])],
            'BMI':                      [float(vals['bmi'])],
            'DiabetesPedigreeFunction': [float(vals['dpf'])],
            'Age':                      [float(vals['age'])],
        })
        # Scale input using the saved scaler
        user_input_scaled = scaler.transform(user_input)
        result = model.predict(user_input_scaled)[0]
        prediction = 'DIABETIC' if result == 1 else 'NOT DIABETIC'
        return render_template_string(HTML_TEMPLATE, prediction=prediction, vals=vals)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f'Error: {str(e)}', vals={})

if __name__ == '__main__':
    app.run(debug=True)
