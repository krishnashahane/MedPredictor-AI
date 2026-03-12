"""ML model training and evaluation for MedPredict AI."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def get_models():
    """Return a dict of model name -> model instance."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and return evaluation results.

    Returns: list of dicts with model name, metrics, and trained model.
    """
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "trained_model": model,
        }
        results.append(metrics)

    return sorted(results, key=lambda x: x["roc_auc"], reverse=True)


def get_best_model(results):
    """Return the best model based on ROC AUC score."""
    return results[0]


def save_model(model, scaler, filename):
    """Save a trained model and scaler to disk."""
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump({"model": model, "scaler": scaler}, filepath)
    return filepath


def load_model(filename):
    """Load a trained model and scaler from disk."""
    filepath = os.path.join(MODELS_DIR, filename)
    data = joblib.load(filepath)
    return data["model"], data["scaler"]
