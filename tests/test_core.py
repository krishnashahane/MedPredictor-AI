import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import models
from data_preprocessing import load_diabetes_data, load_heart_data
from feature_engineering import analyze_features
from models import save_model
from predict import predict_diabetes


def test_datasets_load_from_repository_paths():
    diabetes = load_diabetes_data()
    heart = load_heart_data()
    assert len(diabetes) > 700
    assert len(heart) > 3000
    assert "Outcome" in diabetes
    assert "HeartDiseaseRisk" in heart


def test_feature_analysis_handles_zero_information_mi():
    X = np.zeros((20, 2))
    X[:, 0] = np.arange(20)
    y = np.array([0, 1] * 10)
    analysis = analyze_features(X, y, ["signal", "constant"])
    assert np.isfinite(analysis["combined_score"]).all()


def test_prediction_rejects_non_finite_patient_values():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    patient = {
        "Pregnancies": 1,
        "Glucose": math.nan,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25,
        "DiabetesPedigreeFunction": 0.3,
        "Age": 30,
    }
    try:
        predict_diabetes(model, scaler, patient)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite patient input must be rejected")


def test_model_artifacts_stay_inside_outputs(tmp_path):
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    scaler = StandardScaler().fit([[0], [1]])
    original = models.MODELS_DIR
    models.MODELS_DIR = tmp_path
    try:
        saved = save_model(model, scaler, "safe.joblib")
        assert saved.parent == tmp_path
        assert saved.exists()
    finally:
        models.MODELS_DIR = original
