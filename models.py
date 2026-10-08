"""Model training, evaluation, and trusted artifact persistence."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "outputs"
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def get_models():
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
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(
                    y_test, y_pred, zero_division=0
                ),
                "trained_model": model,
            }
        )
    return sorted(results, key=lambda item: item["roc_auc"], reverse=True)


def get_best_model(results):
    if not results:
        raise ValueError("No model evaluation results were produced.")
    return results[0]


def save_model(model, scaler, filename: str, imputer=None) -> Path:
    """Write a trusted local model artifact inside outputs/."""
    if not _SAFE_FILENAME.fullmatch(filename):
        raise ValueError("Invalid model filename.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump({"model": model, "scaler": scaler, "imputer": imputer}, path)
    return path
