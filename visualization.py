"""Plot generation for the MedPredictor-AI pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _output_path(filename: str) -> Path:
    if not _SAFE_FILENAME.fullmatch(filename):
        raise ValueError("Invalid output filename.")
    return OUTPUT_DIR / filename


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {"figure.figsize": (10, 6), "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12}
    )


def plot_correlation_heatmap(df, title, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation Heatmap - {title}")
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importance_df, title, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Feature Importance - {title}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results, title, filename):
    set_style()
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    plot_df = pd.DataFrame(
        [{"Model": r["model"], "Metric": metric, "Score": r[metric]}
         for r in results for metric in metrics]
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title(f"Model Comparison - {title}")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(results, X_test, y_test, title, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results:
        y_prob = result["trained_model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(
            fpr, tpr, linewidth=2,
            label=f'{result["model"]} (AUC = {auc(fpr, tpr):.3f})',
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {title}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, model_name, title, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name} ({title})")
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(df, target_col, title, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df[target_col].value_counts().sort_index()
    values = [counts.get(0, 0), counts.get(1, 0)]
    ax.bar(
        ["Negative (0)", "Positive (1)"],
        values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
    )
    for index, value in enumerate(values):
        ax.text(index, value + max(max(values) * 0.01, 1), str(value), ha="center")
    ax.set_title(f"Target Distribution - {title}")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = _output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
