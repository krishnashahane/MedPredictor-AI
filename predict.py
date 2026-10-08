"""Prediction helpers for trained MedPredictor-AI research models."""

from __future__ import annotations

import math

import numpy as np

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

HEART_FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]


def _predict(model, scaler, patient_data, features, *, imputer=None, threshold=0.5):
    missing = [feature for feature in features if feature not in patient_data]
    if missing:
        raise ValueError(f"Missing patient fields: {missing}")

    try:
        values = [float(patient_data[feature]) for feature in features]
    except (TypeError, ValueError) as exc:
        raise ValueError("All patient fields must be numeric.") from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Patient values must be finite numbers.")

    X = np.asarray(values, dtype=float).reshape(1, -1)
    if imputer is not None:
        X = imputer.transform(X)
    X_scaled = scaler.transform(X)

    probabilities = np.asarray(model.predict_proba(X_scaled)[0], dtype=float)
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    prediction = int(probabilities[1] >= threshold)
    return {
        "prediction": prediction,
        "threshold": threshold,
        "confidence": float(probabilities.max()) * 100,
        "probability_negative": float(probabilities[0]) * 100,
        "probability_positive": float(probabilities[1]) * 100,
    }


def predict_diabetes(model, scaler, patient_data, *, imputer=None, threshold=0.5):
    result = _predict(
        model, scaler, patient_data, DIABETES_FEATURES,
        imputer=imputer, threshold=threshold,
    )
    result["label"] = "Diabetic" if result["prediction"] else "Non-Diabetic"
    return result


def predict_heart_disease(model, scaler, patient_data, *, imputer=None, threshold=0.5):
    result = _predict(
        model, scaler, patient_data, HEART_FEATURES,
        imputer=imputer, threshold=threshold,
    )
    result["label"] = "Higher Risk" if result["prediction"] else "Lower Risk"
    return result


def get_sample_diabetes_patient():
    return {
        "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.127, "Age": 47,
    }


def get_sample_heart_patient():
    return {
        "male": 1, "age": 55, "education": 2, "currentSmoker": 1,
        "cigsPerDay": 15, "BPMeds": 0, "prevalentStroke": 0,
        "prevalentHyp": 1, "diabetes": 0, "totChol": 250,
        "sysBP": 140, "diaBP": 90, "BMI": 28.5, "heartRate": 75,
        "glucose": 90,
    }


def _interactive_input(prompts):
    data = {}
    for feature, prompt in prompts.items():
        while True:
            try:
                value = float(input(prompt))
                if not math.isfinite(value):
                    raise ValueError
                data[feature] = value
                break
            except ValueError:
                print("  Please enter a finite number.")
    return data


def interactive_diabetes_input():
    print("\n--- Enter Example Patient Data for Diabetes Prediction ---")
    return _interactive_input({
        "Pregnancies": "Number of pregnancies: ",
        "Glucose": "Glucose level (mg/dL): ",
        "BloodPressure": "Blood pressure (mm Hg): ",
        "SkinThickness": "Skin thickness (mm): ",
        "Insulin": "Insulin level (mu U/ml): ",
        "BMI": "BMI: ",
        "DiabetesPedigreeFunction": "Diabetes pedigree function: ",
        "Age": "Age: ",
    })


def interactive_heart_input():
    print("\n--- Enter Example Patient Data for Heart Risk Prediction ---")
    return _interactive_input({
        "male": "Sex (1=male, 0=female): ",
        "age": "Age: ",
        "education": "Education level (1-4): ",
        "currentSmoker": "Current smoker (1=yes, 0=no): ",
        "cigsPerDay": "Cigarettes per day: ",
        "BPMeds": "Blood-pressure medication (1=yes, 0=no): ",
        "prevalentStroke": "History of stroke (1=yes, 0=no): ",
        "prevalentHyp": "Hypertension (1=yes, 0=no): ",
        "diabetes": "Diabetes (1=yes, 0=no): ",
        "totChol": "Total cholesterol (mg/dL): ",
        "sysBP": "Systolic BP (mm Hg): ",
        "diaBP": "Diastolic BP (mm Hg): ",
        "BMI": "BMI: ",
        "heartRate": "Heart rate (bpm): ",
        "glucose": "Glucose level (mg/dL): ",
    })
