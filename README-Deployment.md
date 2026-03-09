# Diabetes Prediction System

🔗 **Live App:** https://Zohaa.pythonanywhere.com

---

## Overview
A machine learning web application that predicts whether a female patient is **Diabetic or Not Diabetic** based on 8 medical measurements. Built using an MLP Neural Network with StandardScaler preprocessing, trained on the Pima Indians Diabetes Database.

## Features
- Predicts diabetes risk from 8 diagnostic inputs
- 80% accuracy on test data
- Live web interface — accessible from any device

## Tech Stack
- **Language:** Python
- **Algorithm:** MLP Neural Network + StandardScaler
- **Libraries:** scikit-learn, pandas, numpy, Flask, pickle
- **Dataset:** Pima Indians Diabetes Database (UCI / Kaggle)
- **Deployment:** PythonAnywhere

## Input Features
| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose level (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years (min. 21) |

## Note
This model is trained on data from **females aged 21 and above** of Pima Indian heritage. Predictions are only valid for this demographic. For educational purposes only — always consult a medical professional for diagnosis.