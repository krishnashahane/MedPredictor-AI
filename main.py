#!/usr/bin/env python3
"""
MedPredict AI - Main Entry Point
AI-powered healthcare intelligence system for disease prediction.
"""

import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore")

from src.data_preprocessing import (
    load_diabetes_data,
    load_heart_data,
    prepare_dataset,
    get_dataset_summary,
)
from src.feature_engineering import analyze_features, get_feature_importance
from src.models import get_models, train_and_evaluate, get_best_model, save_model
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_target_distribution,
)
from src.predict import (
    predict_diabetes,
    predict_heart_disease,
    get_sample_diabetes_patient,
    get_sample_heart_patient,
    interactive_diabetes_input,
    interactive_heart_input,
)


def print_header():
    print("=" * 60)
    print("  🧠🩺 MedPredict AI - Disease Prediction System")
    print("=" * 60)
    print()


def print_section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def print_metrics(results):
    """Print model evaluation metrics in a table."""
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC AUC':>10}")
    print(f"  {'─' * 75}")
    for r in results:
        print(
            f"  {r['model']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f}"
        )
    print()


def run_pipeline(disease, interactive=False):
    """Run the full ML pipeline for a disease."""

    if disease == "diabetes":
        print_section("Loading Diabetes Dataset (PIMA Indians)")
        df = load_diabetes_data()
        target_col = "Outcome"
        disease_title = "Diabetes"
        model_filename = "diabetes_model.pkl"
    elif disease == "heart":
        print_section("Loading Heart Disease Dataset (Framingham)")
        df = load_heart_data()
        target_col = "HeartDiseaseRisk"
        disease_title = "Heart Disease"
        model_filename = "heart_model.pkl"
    else:
        print(f"  Unknown disease: {disease}")
        return

    # Dataset summary
    summary = get_dataset_summary(df, disease_title)
    print(f"  Rows: {summary['rows']}  |  Columns: {summary['columns']}  |  Missing: {summary['missing_values']}")
    print(f"  Features: {', '.join(summary['features'])}")

    # Target distribution
    print_section(f"Target Distribution - {disease_title}")
    target_counts = df[target_col].value_counts()
    total = len(df)
    print(f"  Negative (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Positive (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/total*100:.1f}%)")

    # Visualize target distribution
    plot_target_distribution(df, target_col, disease_title, f"{disease}_target_dist.png")

    # Prepare data
    print_section("Preparing Data")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_dataset(df, target_col)
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")

    # Feature analysis
    print_section("Feature Analysis")
    importance = get_feature_importance(X_train, y_train, feature_names)
    analysis = analyze_features(X_train, y_train, feature_names)
    print("  Top features by combined score:")
    for _, row in analysis.head(5).iterrows():
        print(f"    • {row['feature']}: importance={row['importance']:.4f}, MI={row['mi_score']:.4f}")

    # Visualizations
    print_section("Generating Visualizations")
    p1 = plot_correlation_heatmap(df, disease_title, f"{disease}_correlation.png")
    print(f"  ✓ Correlation heatmap: {p1}")
    p2 = plot_feature_importance(importance, disease_title, f"{disease}_feature_importance.png")
    print(f"  ✓ Feature importance:  {p2}")

    # Train models
    print_section("Training Models")
    models = get_models()
    print(f"  Training {len(models)} models: {', '.join(models.keys())}")
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # Results
    print_section(f"Model Performance - {disease_title}")
    print_metrics(results)

    best = get_best_model(results)
    print(f"  🏆 Best Model: {best['model']} (ROC AUC: {best['roc_auc']:.4f})")

    # More visualizations
    p3 = plot_model_comparison(results, disease_title, f"{disease}_model_comparison.png")
    print(f"  ✓ Model comparison: {p3}")
    p4 = plot_roc_curves(results, X_test, y_test, disease_title, f"{disease}_roc_curves.png")
    print(f"  ✓ ROC curves:       {p4}")
    p5 = plot_confusion_matrix(
        best["confusion_matrix"], best["model"], disease_title,
        f"{disease}_confusion_matrix.png"
    )
    print(f"  ✓ Confusion matrix: {p5}")

    # Save best model
    print_section("Saving Best Model")
    model_path = save_model(best["trained_model"], scaler, model_filename)
    print(f"  ✓ Model saved: {model_path}")

    # Prediction demo
    print_section("Prediction Demo")
    if interactive:
        if disease == "diabetes":
            patient = interactive_diabetes_input()
        else:
            patient = interactive_heart_input()
    else:
        if disease == "diabetes":
            patient = get_sample_diabetes_patient()
        else:
            patient = get_sample_heart_patient()

    print(f"\n  Patient Data: {patient}")

    if disease == "diabetes":
        result = predict_diabetes(best["trained_model"], scaler, patient)
    else:
        result = predict_heart_disease(best["trained_model"], scaler, patient)

    print(f"\n  🔮 Prediction: {result['label']}")
    print(f"  📊 Confidence: {result['confidence']:.1f}%")
    print(f"     Negative probability: {result['probability_negative']:.1f}%")
    print(f"     Positive probability: {result['probability_positive']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MedPredict AI - Disease Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Run full pipeline for all diseases
  python main.py --disease diabetes  Run only diabetes prediction
  python main.py --disease heart     Run only heart disease prediction
  python main.py --interactive       Enable interactive patient input
        """,
    )
    parser.add_argument(
        "--disease",
        choices=["diabetes", "heart", "all"],
        default="all",
        help="Which disease to predict (default: all)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive patient data input",
    )

    args = parser.parse_args()
    print_header()

    diseases = ["diabetes", "heart"] if args.disease == "all" else [args.disease]

    for disease in diseases:
        run_pipeline(disease, interactive=args.interactive)
        print()

    print("=" * 60)
    print("  ✅ MedPredict AI pipeline complete!")
    print(f"  📁 Results saved to: {os.path.join(os.path.dirname(__file__), 'outputs')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
