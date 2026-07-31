# src/train.py
# Model training module for Telco Customer Churn prediction

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(PROJECT_ROOT, "data", "cleaned_churn.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
FEATURE_IMP_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.json")


def load_data(path: str):
    """Load cleaned dataset and split into features and target."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"[train] Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Perform train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[train] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """Initialize and train a RandomForestClassifier."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    print(f"[train] Model trained with {n_estimators} estimators")
    return clf


def evaluate_model(clf, X_test, y_test):
    """Compute and print key classification metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\n" + "=" * 40)
    print("  MODEL EVALUATION METRICS")
    print("=" * 40)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("=" * 40 + "\n")
    return {"accuracy": round(acc, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def save_model(clf, path: str):
    """Serialize and save the trained model with joblib."""
    joblib.dump(clf, path)
    print(f"[train] Model saved to {path}")


def save_feature_importance(clf, feature_names, path: str):
    """Extract and save feature importances to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    importances = clf.feature_importances_.tolist()
    importance_dict = dict(zip(feature_names, importances))
    # Sort descending
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    with open(path, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"[train] Feature importance saved to {path}")
    top3 = list(importance_dict.items())[:3]
    print(f"[train] Top 3 features: {top3}")


def run_training_pipeline():
    """Full end-to-end training pipeline."""
    X, y = load_data(CLEAN_CSV)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_model(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    save_model(clf, MODEL_PATH)
    save_feature_importance(clf, X.columns.tolist(), FEATURE_IMP_PATH)
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
