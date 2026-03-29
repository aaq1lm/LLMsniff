#!/usr/bin/env python3
"""
classifier.py — Train and evaluate Random Forest classifiers for the
LLM token-length side-channel attack.

ARCHITECTURAL NOTE:
This script now runs the entire evaluation suite TWICE:
  1. Oracle Baseline (Chunk Features): Proves the theoretical maximum leakage
     before TLS encryption and network framing.
  2. Real-World Attacker (Packet Features): Proves the practical vulnerability
     using only passive network metadata.

VALIDATION:
Uses Nested Cross-Validation. A 3-fold GridSearchCV is performed inside a 
5-fold StratifiedKFold. This is computationally expensive but provides the
most rigorous, academically defensible metrics for 2,000 samples.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES_FILE = "features.csv"
RESULTS_FILE = "results_summary.csv"
N_FOLDS = 5
RANDOM_STATE = 42


def get_feature_columns(df: pd.DataFrame, feature_type: str) -> list:
    """
    Dynamically select either chunk or packet features based on prefixes.
    """
    if feature_type == "chunk":
        return [c for c in df.columns if c.startswith("chunk_") or c in ["total_chunks", "total_bytes"]]
    elif feature_type == "packet":
        return [c for c in df.columns if c.startswith("packet_") or c == "total_packets"]
    else:
        raise ValueError("feature_type must be 'chunk' or 'packet'")


def run_cv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list,
    classifier_name: str,
    confusion_matrix_file: str,
    feature_names: list,
) -> dict:
    """
    Run 5-fold Stratified Cross-Validation with a Random Forest classifier.
    """
    print(f"\n{'=' * 70}")
    print(f"  Classifier: {classifier_name}")
    print(f"  Samples: {len(y)} | Classes: {len(label_names)} | Features: {X.shape[1]}")
    print(f"  Validation: {N_FOLDS}-fold Stratified CV (with Nested GridSearch)")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    fold_f1_macros = []
    all_y_true = []
    all_y_pred = []
    all_importances = np.zeros(X.shape[1])

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        
        # n_jobs=-1 utilizes all available CPU cores for the grid search
        base_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        grid_search = GridSearchCV(
            base_clf, param_grid, cv=3, scoring="f1_macro", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_
        
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_accuracies.append(acc)
        fold_f1_macros.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_importances += clf.feature_importances_

        print(f"  Fold {fold_idx}: accuracy={acc:.4f}  macro-F1={f1:.4f}  (Best Params: {grid_search.best_params_})")

    # Averaged metrics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_macros)
    std_f1 = np.std(fold_f1_macros)

    print(f"\n  Average Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Average Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # Aggregated classification report (across all folds)
    print(f"\n  Aggregated Classification Report:")
    report = classification_report(
        all_y_true, all_y_pred, target_names=label_names, zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(label_names)), max(6, len(label_names) * 0.8)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{classifier_name}\nConfusion Matrix ({N_FOLDS}-fold CV)", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(confusion_matrix_file, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix → {confusion_matrix_file}")

    # Top 10 most important features
    avg_importances = all_importances / N_FOLDS
    top_indices = np.argsort(avg_importances)[::-1][:10]
    print(f"\n  Top 10 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:<25s} importance={avg_importances[idx]:.4f}")

    # Save ALL feature importances to CSV for visualizer.py to use
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_importances,
        "classifier": classifier_name,
    }).sort_values("importance", ascending=False)
    safe_name = classifier_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    importance_file = f"feature_importance_{safe_name}.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"  Saved feature importances → {importance_file}")


    # Per-class F1 from aggregated report
    report_dict = classification_report(
        all_y_true, all_y_pred, target_names=label_names,
        output_dict=True, zero_division=0,
    )

    return {
        "classifier": classifier_name,
        "n_samples": len(y),
        "n_classes": len(label_names),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "per_class_f1": {name: report_dict[name]["f1-score"] for name in label_names},
    }


def main():
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"[FATAL] {FEATURES_FILE} not found. Run feature_extractor.py first.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(df)} rows from {FEATURES_FILE}")

    results = []

    # Loop over both feature sets to compare Theoretical vs Practical attacks
    for feature_type in ["chunk", "packet"]:
        
        feature_cols = get_feature_columns(df, feature_type)
        print(f"\n[INFO] Initializing {feature_type.upper()} test suite with {len(feature_cols)} features...")
        
        prefix = "Oracle (Chunk)" if feature_type == "chunk" else "Attacker (Packet)"

        # ---- Classifier 1: Binary sensitivity (High vs Low) ----
        y_binary = df["sensitivity"].values
        X_binary = df[feature_cols].values
        binary_result = run_cv_classifier(
            X_binary, y_binary,
            label_names=["High", "Low"],
            classifier_name=f"{prefix} - Binary Sensitivity",
            confusion_matrix_file=f"confusion_matrix_binary_{feature_type}.png",
            feature_names=feature_cols,
        )
        results.append(binary_result)

        # ---- Classifier 2: 10-class domain ----
        le_domain = LabelEncoder()
        y_domain = le_domain.fit_transform(df["category"].values)
        X_domain = df[feature_cols].values
        domain_result = run_cv_classifier(
            X_domain, y_domain,
            label_names=list(le_domain.classes_),
            classifier_name=f"{prefix} - 10-Class Domain",
            confusion_matrix_file=f"confusion_matrix_10class_{feature_type}.png",
            feature_names=feature_cols,
        )
        results.append(domain_result)

        # ---- Classifier 3: Sensitive-only 6-class ----
        sensitive_df = df[df["sensitivity"] == "High"].copy()
        le_sensitive = LabelEncoder()
        y_sensitive = le_sensitive.fit_transform(sensitive_df["category"].values)
        X_sensitive = sensitive_df[feature_cols].values
        sensitive_result = run_cv_classifier(
            X_sensitive, y_sensitive,
            label_names=list(le_sensitive.classes_),
            classifier_name=f"{prefix} - Sensitive-Only",
            confusion_matrix_file=f"confusion_matrix_sensitive_{feature_type}.png",
            feature_names=feature_cols,
        )
        results.append(sensitive_result)

    # ---- Timing-Only Attack: Proves timing alone is sufficient ----
    # This backs up the "Timing-Optimized Attack" claim in the paper.
    # Uses ONLY inter-chunk/inter-packet timing features — zero size information.
    # If this achieves high accuracy, it confirms that byte padding cannot defeat
    # the attack because it leaves the timing signal completely intact.
    timing_cols = [
        c for c in df.columns
        if c in ["chunk_ict_mean", "chunk_ict_std", "chunk_ict_max",
                 "packet_ipt_mean", "packet_ipt_std", "packet_ipt_max"]
    ]
    print(f"\n[INFO] Timing-Only Attack using {len(timing_cols)} features: {timing_cols}")

    y_binary_timing = df["sensitivity"].values
    X_timing = df[timing_cols].values
    timing_result = run_cv_classifier(
        X_timing, y_binary_timing,
        label_names=["High", "Low"],
        classifier_name="Timing-Only Attack - Binary Sensitivity",
        confusion_matrix_file="confusion_matrix_timing_only.png",
        feature_names=timing_cols,
    )
    results.append(timing_result)

    # ---- Save results summary ----
    summary_rows = []
    for r in results:
        row = {
            "classifier": r["classifier"],
            "n_samples": r["n_samples"],
            "n_classes": r["n_classes"],
            "mean_accuracy": round(r["mean_accuracy"], 4),
            "std_accuracy": round(r["std_accuracy"], 4),
            "mean_macro_f1": round(r["mean_macro_f1"], 4),
            "std_macro_f1": round(r["std_macro_f1"], 4),
        }
        for class_name, f1_val in r["per_class_f1"].items():
            row[f"f1_{class_name}"] = round(f1_val, 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n[OK] Saved comprehensive results summary to {RESULTS_FILE}")


if __name__ == "__main__":
    main()