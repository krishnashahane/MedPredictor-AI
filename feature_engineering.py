"""Feature analysis and selection for MedPredict AI."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def get_feature_importance(X_train, y_train, feature_names):
    """Calculate feature importance using Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    return importance


def get_mutual_information(X_train, y_train, feature_names):
    """Calculate mutual information scores for features."""
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    mi_df = pd.DataFrame({
        "feature": feature_names,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False)

    return mi_df


def get_correlation_matrix(df):
    """Return the correlation matrix of the dataframe."""
    return df.corr(numeric_only=True)


def analyze_features(X_train, y_train, feature_names):
    """Run full feature analysis and return results."""
    importance = get_feature_importance(X_train, y_train, feature_names)
    mi_scores = get_mutual_information(X_train, y_train, feature_names)

    # Merge results
    analysis = importance.merge(mi_scores, on="feature")
    analysis["combined_score"] = (
        analysis["importance"] / analysis["importance"].max()
        + analysis["mi_score"] / analysis["mi_score"].max()
    ) / 2

    return analysis.sort_values("combined_score", ascending=False)
