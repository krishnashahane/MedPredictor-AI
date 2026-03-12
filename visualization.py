"""Data visualization for MedPredict AI."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_style():
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_correlation_heatmap(df, title, filename):
    """Plot and save correlation heatmap."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax
    )
    ax.set_title(f"Correlation Heatmap - {title}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importance_df, title, filename):
    """Plot and save feature importance bar chart."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance - {title}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results, title, filename):
    """Plot model comparison bar chart."""
    set_style()
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = [r["model"] for r in results]

    data = []
    for r in results:
        for m in metrics_to_plot:
            data.append({"Model": r["model"], "Metric": m, "Score": r[m]})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title(f"Model Comparison - {title}")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(results, X_test, y_test, title, filename):
    """Plot ROC curves for all models."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        model = r["trained_model"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{r["model"]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {title}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, model_name, title, filename):
    """Plot confusion matrix heatmap."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name} ({title})")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(df, target_col, title, filename):
    """Plot target variable distribution."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df[target_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=colors, edgecolor="black")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title(f"Target Distribution - {title}")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
