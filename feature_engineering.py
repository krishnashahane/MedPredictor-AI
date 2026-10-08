"""Feature analysis for MedPredictor-AI."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


def get_feature_importance(X_train, y_train, feature_names) -> pd.DataFrame:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)


def get_mutual_information(X_train, y_train, feature_names) -> pd.DataFrame:
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    return pd.DataFrame(
        {"feature": feature_names, "mi_score": scores}
    ).sort_values("mi_score", ascending=False)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


def analyze_features(X_train, y_train, feature_names) -> pd.DataFrame:
    importance = get_feature_importance(X_train, y_train, feature_names)
    mi_scores = get_mutual_information(X_train, y_train, feature_names)
    analysis = importance.merge(mi_scores, on="feature")

    importance_max = float(analysis["importance"].max())
    mi_max = float(analysis["mi_score"].max())
    importance_norm = (
        analysis["importance"] / importance_max if importance_max > 0 else 0.0
    )
    mi_norm = analysis["mi_score"] / mi_max if mi_max > 0 else 0.0
    analysis["combined_score"] = (importance_norm + mi_norm) / 2
    return analysis.sort_values("combined_score", ascending=False)
