"""Model selection, evaluation, threshold tuning, and trusted artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "outputs"
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def get_models(include_xgboost: bool = True, fast: bool = False):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2500, random_state=42, solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "MLP": MLPClassifier(max_iter=1200, early_stopping=True, random_state=42),
    }

    if not fast and include_xgboost:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=2,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                reg_alpha=0.05,
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            pass

    return models


def _search_space(name: str):
    return {
        "Logistic Regression": {
            "C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "n_estimators": [300, 500],
            "max_depth": [None, 6, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
        },
        "Extra Trees": {
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [1, 3, 5],
        },
        "Hist Gradient Boosting": {
            "max_iter": [150, 250, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_leaf_nodes": [7, 15, 31],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        "SVM": {
            "C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
            "class_weight": [None, "balanced"],
        },
        "MLP": {
            "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [0.0005, 0.001, 0.003],
        },
        "XGBoost": {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.02, 0.03, 0.05],
            "max_depth": [2, 3, 4],
            "min_child_weight": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        },
    }.get(name, {})


def _tune_model(name, model, X_train, y_train, cv, fast):
    space = _search_space(name)
    if fast or not space:
        model.fit(X_train, y_train)
        return model, {
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
        }

    search = RandomizedSearchCV(
        model,
        space,
        n_iter=8,
        scoring="accuracy",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, {
        "cv_accuracy_mean": float(search.best_score_),
        "cv_accuracy_std": float(search.cv_results_["std_test_score"][search.best_index_]),
    }


def _find_accuracy_threshold(y_true, probabilities, min_recall=0.50):
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.linspace(0.10, 0.90, 161):
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        accuracy = accuracy_score(y_true, predictions)
        if recall >= min_recall and accuracy > best_accuracy:
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
    return best_threshold


def train_and_evaluate(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    fast=False,
    min_recall=0.50,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        tuned_model, cv_metrics = _tune_model(
            name, model, X_train, y_train, cv, fast
        )

        threshold = 0.5
        if not fast:
            oof_probabilities = cross_val_predict(
                tuned_model,
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            threshold = _find_accuracy_threshold(
                y_train, oof_probabilities, min_recall=min_recall
            )

        probabilities = tuned_model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(y_test, probabilities),
                "threshold": threshold,
                "confusion_matrix": confusion_matrix(y_test, predictions),
                "classification_report": classification_report(
                    y_test, predictions, zero_division=0
                ),
                "trained_model": tuned_model,
                **cv_metrics,
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item["cv_accuracy_mean"]
            if np.isfinite(item["cv_accuracy_mean"])
            else item["accuracy"],
            item["balanced_accuracy"],
            item["roc_auc"],
        ),
        reverse=True,
    )


def get_best_model(results):
    if not results:
        raise ValueError("No model evaluation results were produced.")
    return results[0]


def save_model(
    model,
    scaler,
    filename: str,
    imputer=None,
    threshold: float = 0.5,
) -> Path:
    """Write a trusted local model artifact inside outputs/."""
    if not _SAFE_FILENAME.fullmatch(filename):
        raise ValueError("Invalid model filename.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "threshold": float(threshold),
        },
        path,
    )
    return path
