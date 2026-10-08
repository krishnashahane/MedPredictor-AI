#!/usr/bin/env python3
"""MedPredictor-AI training, benchmarking, and prediction CLI."""

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
TARGET_ACCURACY = 0.90


def print_header():
    print("=" * 72)
    print("  MedPredictor-AI - Research Model Benchmark")
    print("=" * 72)


def print_section(title):
    print(f"\n{'─' * 56}\n  {title}\n{'─' * 56}")


def print_metrics(results):
    print(
        f"\n  {'Model':<23} {'CV Acc':>8} {'Test Acc':>9} "
        f"{'Bal Acc':>9} {'Recall':>8} {'ROC AUC':>9}"
    )
    print(f"  {'─' * 70}")
    for result in results:
        cv = result["cv_accuracy_mean"]
        cv_text = f"{cv:.3f}" if cv == cv else "n/a"
        print(
            f"  {result['model']:<23} {cv_text:>8} "
            f"{result['accuracy']:>9.3f} {result['balanced_accuracy']:>9.3f} "
            f"{result['recall']:>8.3f} {result['roc_auc']:>9.3f}"
        )
    print()


def run_pipeline(
    disease,
    interactive=False,
    fast=False,
    no_xgboost=False,
):
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

    print_section(f"{disease_title}: data and split")
    summary = get_dataset_summary(df, disease_title)
    print(
        f"  Rows: {summary['rows']} | Columns: {summary['columns']} "
        f"| Missing after loading: {summary['missing_values']}"
    )
    plot_target_distribution(df, target_col, disease_title, f"{disease}_target_dist.png")

    X_train, X_test, y_train, y_test, scaler, feature_names, imputer = prepare_dataset(
        df, target_col
    )
    print(f"  Training rows: {len(y_train)}")
    print(f"  Held-out test rows: {len(y_test)}")

    print_section("Feature analysis")
    importance = get_feature_importance(X_train, y_train, feature_names)
    analysis = analyze_features(X_train, y_train, feature_names)
    for _, row in analysis.head(5).iterrows():
        print(
            f"    • {row['feature']}: importance={row['importance']:.4f}, "
            f"MI={row['mi_score']:.4f}"
        )

    plot_correlation_heatmap(df, disease_title, f"{disease}_correlation.png")
    plot_feature_importance(importance, disease_title, f"{disease}_feature_importance.png")

    print_section("Model search")
    models = get_models(include_xgboost=not no_xgboost, fast=fast)
    print(f"  Candidates: {', '.join(models)}")
    if not fast:
        print("  Selection: 5-fold stratified CV on training data")
        print("  Final metric: one untouched 20% test split")
        print("  Threshold tuning: out-of-fold accuracy with minimum recall 0.50")

    results = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        fast=fast,
        min_recall=0.50,
    )

    print_section("Benchmark")
    print_metrics(results)
    best = get_best_model(results)
    target_met = best["accuracy"] >= TARGET_ACCURACY
    print(f"  Best model: {best['model']}")
    print(f"  Decision threshold: {best['threshold']:.3f}")
    print(
        f"  90% held-out accuracy target: "
        f"{'MET' if target_met else 'NOT MET'} ({best['accuracy'] * 100:.2f}%)"
    )
    if not target_met:
        print(
            "  The test set is untouched; do not manipulate it to force the target. "
            "Use better data or external validation instead."
        )

    plot_model_comparison(results, disease_title, f"{disease}_model_comparison.png")
    plot_roc_curves(results, X_test, y_test, disease_title, f"{disease}_roc_curves.png")
    plot_confusion_matrix(
        best["confusion_matrix"],
        best["model"],
        disease_title,
        f"{disease}_confusion_matrix.png",
    )

    model_path = save_model(
        best["trained_model"],
        scaler,
        model_filename,
        imputer=imputer,
        threshold=best["threshold"],
    )
    print(f"  Saved trusted model: {model_path}")

    print_section("Prediction demo")
    patient = (
        interactive_diabetes_input()
        if interactive and disease == "diabetes"
        else interactive_heart_input()
        if interactive
        else get_sample_diabetes_patient()
        if disease == "diabetes"
        else get_sample_heart_patient()
    )

    result = (
        predict_diabetes(
            best["trained_model"],
            scaler,
            patient,
            imputer=imputer,
            threshold=best["threshold"],
        )
        if disease == "diabetes"
        else predict_heart_disease(
            best["trained_model"],
            scaler,
            patient,
            imputer=imputer,
            threshold=best["threshold"],
        )
    )
    print(f"  Prediction: {result['label']}")
    print(f"  Positive probability: {result['probability_positive']:.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and run MedPredictor-AI research models."
    )
    parser.add_argument("--disease", choices=["diabetes", "heart", "all"], default="all")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip hyperparameter search for quick smoke tests.",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Disable optional XGBoost even when installed.",
    )
    args = parser.parse_args()

    print_header()
    diseases = ["diabetes", "heart"] if args.disease == "all" else [args.disease]
    for disease in diseases:
        run_pipeline(
            disease,
            interactive=args.interactive,
            fast=args.fast,
            no_xgboost=args.no_xgboost,
        )
    print(f"\nOutputs: {ROOT_DIR / 'outputs'}")


if __name__ == "__main__":
    raise SystemExit(main())
