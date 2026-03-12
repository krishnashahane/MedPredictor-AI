"""Prediction engine for MedPredict AI."""

import numpy as np
import pandas as pd


DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

HEART_FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]


def predict_diabetes(model, scaler, patient_data):
    """Predict diabetes risk for a patient.

    Args:
        model: trained model
        scaler: fitted StandardScaler
        patient_data: dict with keys matching DIABETES_FEATURES

    Returns: dict with prediction and probability
    """
    values = [patient_data[f] for f in DIABETES_FEATURES]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "confidence": float(max(probability)) * 100,
        "probability_negative": float(probability[0]) * 100,
        "probability_positive": float(probability[1]) * 100,
    }


def predict_heart_disease(model, scaler, patient_data):
    """Predict heart disease risk for a patient.

    Args:
        model: trained model
        scaler: fitted StandardScaler
        patient_data: dict with keys matching HEART_FEATURES

    Returns: dict with prediction and probability
    """
    values = [patient_data[f] for f in HEART_FEATURES]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": float(max(probability)) * 100,
        "probability_negative": float(probability[0]) * 100,
        "probability_positive": float(probability[1]) * 100,
    }


def get_sample_diabetes_patient():
    """Return a sample patient for diabetes prediction demo."""
    return {
        "Pregnancies": 2,
        "Glucose": 138,
        "BloodPressure": 62,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.127,
        "Age": 47,
    }


def get_sample_heart_patient():
    """Return a sample patient for heart disease prediction demo."""
    return {
        "male": 1,
        "age": 55,
        "education": 2,
        "currentSmoker": 1,
        "cigsPerDay": 15,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 250,
        "sysBP": 140,
        "diaBP": 90,
        "BMI": 28.5,
        "heartRate": 75,
        "glucose": 90,
    }


def interactive_diabetes_input():
    """Collect diabetes prediction input from user interactively."""
    print("\n--- Enter Patient Data for Diabetes Prediction ---")
    data = {}
    prompts = {
        "Pregnancies": "Number of pregnancies: ",
        "Glucose": "Glucose level (mg/dL): ",
        "BloodPressure": "Blood Pressure (mm Hg): ",
        "SkinThickness": "Skin Thickness (mm): ",
        "Insulin": "Insulin level (mu U/ml): ",
        "BMI": "BMI: ",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function: ",
        "Age": "Age: ",
    }
    for feature, prompt in prompts.items():
        while True:
            try:
                data[feature] = float(input(prompt))
                break
            except ValueError:
                print("  Please enter a valid number.")
    return data


def interactive_heart_input():
    """Collect heart disease prediction input from user interactively."""
    print("\n--- Enter Patient Data for Heart Disease Prediction ---")
    data = {}
    prompts = {
        "male": "Gender (1=Male, 0=Female): ",
        "age": "Age: ",
        "education": "Education level (1-4): ",
        "currentSmoker": "Current smoker (1=Yes, 0=No): ",
        "cigsPerDay": "Cigarettes per day: ",
        "BPMeds": "On BP medication (1=Yes, 0=No): ",
        "prevalentStroke": "History of stroke (1=Yes, 0=No): ",
        "prevalentHyp": "Hypertension (1=Yes, 0=No): ",
        "diabetes": "Diabetes (1=Yes, 0=No): ",
        "totChol": "Total cholesterol (mg/dL): ",
        "sysBP": "Systolic BP (mm Hg): ",
        "diaBP": "Diastolic BP (mm Hg): ",
        "BMI": "BMI: ",
        "heartRate": "Heart rate (bpm): ",
        "glucose": "Glucose level (mg/dL): ",
    }
    for feature, prompt in prompts.items():
        while True:
            try:
                data[feature] = float(input(prompt))
                break
            except ValueError:
                print("  Please enter a valid number.")
    return data
