#!/usr/bin/env python3
"""MedPredictor-AI command-line training and prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing import (
    get_dataset_summary,
    load_diabetes_data,
    load_heart_data,
    prepare_dataset,
)
from feature_engineering import analyze_features, get_feature_importance
from models import get_best_model, get_models, save_model, train_and_evaluate
from predict import (
    get_sample_diabetes_patient,
    get_sample_heart_patient,
    interactive_diabetes_input,
    interactive_heart_input,
    predict_diabetes,
    predict_heart_disease,
)
from visualization import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    plot_target_distribution,
)

ROOT_DIR = Path(__file__).resolve().parent


def print_header() -> None:
    print("=" * 60)
    print("  MedPredictor-AI - Disease Prediction Research Pipeline")
    print("=" * 60)
    print()


def print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def print_metrics(results: list[dict]) -> None:
    print(
        f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'ROC AUC':>10}"
    )
    print(f"  {'─' * 75}")
    for result in results:
        print(
            f"  {result['model']:<25} {result['accuracy']:>10.4f} "
            f"{result['precision']:>10.4f} {result['recall']:>10.4f} "
            f"{result['f1_score']:>10.4f} {result['roc_auc']:>10.4f}"
        )
    print()


def run_pipeline(disease: str, interactive: bool = False) -> list[dict]:
    """Train, evaluate, visualize, save, and demonstrate one disease pipeline."""
    if disease == "diabetes":
        df = load_diabetes_data()
        target_col = "Outcome"
        disease_title = "Diabetes"
        model_filename = "diabetes_model.joblib"
    elif disease == "heart":
        df = load_heart_data()
        target_col = "HeartDiseaseRisk"
        disease_title = "Heart Disease"
        model_filename = "heart_model.joblib"
    else:
        raise ValueError(f"Unsupported disease: {disease}")

    print_section(f"Loading {disease_title} Dataset")
    summary = get_dataset_summary(df, disease_title)
    print(
        f"  Rows: {summary['rows']} | Columns: {summary['columns']} "
        f"| Missing: {summary['missing_values']}"
    )
    print(f"  Features: {', '.join(summary['features'])}")

    print_section(f"Target Distribution - {disease_title}")
    target_counts = df[target_col].value_counts()
    total = len(df)
    negative = int(target_counts.get(0, 0))
    positive = int(target_counts.get(1, 0))
    print(f"  Negative (0): {negative} ({negative / total * 100:.1f}%)")
    print(f"  Positive (1): {positive} ({positive / total * 100:.1f}%)")
    plot_target_distribution(df, target_col, disease_title, f"{disease}_target_dist.png")

    print_section("Preparing Data")
    X_train, X_test, y_train, y_test, scaler, feature_names, imputer = prepare_dataset(
        df, target_col
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    print_section("Feature Analysis")
    importance = get_feature_importance(X_train, y_train, feature_names)
    analysis = analyze_features(X_train, y_train, feature_names)
    for _, row in analysis.head(5).iterrows():
        print(
            f"    • {row['feature']}: importance={row['importance']:.4f}, "
            f"MI={row['mi_score']:.4f}"
        )

    print_section("Generating Visualizations")
    print(
        "  Correlation heatmap:",
        plot_correlation_heatmap(
            df, disease_title, f"{disease}_correlation.png"
        ),
    )
    print(
        "  Feature importance:",
        plot_feature_importance(
            importance, disease_title, f"{disease}_feature_importance.png"
        ),
    )

    print_section("Training Models")
    models = get_models()
    print(f"  Training {len(models)} models: {', '.join(models)}")
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print_section(f"Model Performance - {disease_title}")
    print_metrics(results)
    best = get_best_model(results)
    print(f"  Best Model: {best['model']} (ROC AUC: {best['roc_auc']:.4f})")

    print(
        "  Model comparison:",
        plot_model_comparison(
            results, disease_title, f"{disease}_model_comparison.png"
        ),
    )
    print(
        "  ROC curves:",
        plot_roc_curves(
            results, X_test, y_test, disease_title, f"{disease}_roc_curves.png"
        ),
    )
    print(
        "  Confusion matrix:",
        plot_confusion_matrix(
            best["confusion_matrix"],
            best["model"],
            disease_title,
            f"{disease}_confusion_matrix.png",
        ),
    )

    print_section("Saving Best Model")
    model_path = save_model(best["trained_model"], scaler, model_filename, imputer=imputer)
    print(f"  Model saved: {model_path}")

    print_section("Prediction Demo")
    patient = (
        interactive_diabetes_input()
        if interactive and disease == "diabetes"
        else interactive_heart_input()
        if interactive
        else get_sample_diabetes_patient()
        if disease == "diabetes"
        else get_sample_heart_patient()
    )

    print(f"\n  Patient Data: {patient}")
    result = (
        predict_diabetes(best["trained_model"], scaler, patient)
        if disease == "diabetes"
        else predict_heart_disease(best["trained_model"], scaler, patient)
    )

    print(f"\n  Prediction: {result['label']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Negative probability: {result['probability_negative']:.1f}%")
    print(f"  Positive probability: {result['probability_positive']:.1f}%")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MedPredictor-AI research models.",
    )
    parser.add_argument(
        "--disease",
        choices=["diabetes", "heart", "all"],
        default="all",
        help="Disease pipeline to run (default: all).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter a custom example patient after model training.",
    )
    args = parser.parse_args()

    print_header()
    diseases = ["diabetes", "heart"] if args.disease == "all" else [args.disease]

    for disease in diseases:
        run_pipeline(disease, interactive=args.interactive)
        print()

    print("=" * 60)
    print(f"  Pipeline complete. Outputs: {ROOT_DIR / 'outputs'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
