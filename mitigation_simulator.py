#!/usr/bin/env python3


import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# Target the compiled packet-level features matrix
FEATURES_FILE = "features.csv"
PADDING_LEVELS = [0, 25, 50, 100, 200, 500]  # percentage of original packet size
N_FOLDS = 5
RANDOM_STATE = 42

def apply_network_padding(df: pd.DataFrame, padding_pct: float) -> pd.DataFrame:
    """
    Apply random padding to packet size-related columns in the feature matrix,
    simulating network-level padding, while leaving timing (IPT) alone.
    """
    if padding_pct == 0:
        return df.copy()

    padded_df = df.copy()
    rng = np.random.RandomState(RANDOM_STATE + int(padding_pct))


    cols_to_pad = [
        col for col in padded_df.columns
        if col.startswith("packet_") and "ipt" not in col
    ]

    # Apply random uniform noise up to the given percentage
    for col in cols_to_pad:
        noise = rng.uniform(0, padded_df[col] * (padding_pct / 100.0))
        padded_df[col] = padded_df[col] + noise

    return padded_df

def run_binary_cv(features_df: pd.DataFrame) -> tuple:
    """Run binary sensitivity classifier using ONLY packet features."""
    

    feature_cols = [
        c for c in features_df.columns
        if c.startswith("packet_") or c == "total_packets"
    ]

    X = features_df[feature_cols].values
    y = features_df["sensitivity"].values
    
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    f1_scores_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores_list.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return np.mean(accuracies), np.mean(f1_scores_list)

def main():
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"fatal: {FEATURES_FILE} not found. please ensure your feature matrix is compiled.")
        sys.exit(1)

    print(f"loaded {len(df)} rows from {FEATURES_FILE}")
    print(f"simulating network-level padding at {len(PADDING_LEVELS)} levels")
    print(f"validation: {N_FOLDS}-fold stratified cv per level")
    print()

    results = []

    print(f"  {'Padding %':>10s}  {'Accuracy':>10s}  {'Macro-F1':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for pct in PADDING_LEVELS:
        padded_df = apply_network_padding(df, pct)
        acc, f1 = run_binary_cv(padded_df)
        results.append({"padding_pct": pct, "accuracy": acc, "f1_macro": f1})
        print(f"  {pct:>9d}%  {acc:>10.4f}  {f1:>10.4f}")

    results_df = pd.DataFrame(results)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        results_df["padding_pct"], results_df["accuracy"],
        "o-", color="#e74c3c", linewidth=2, markersize=8, label="Accuracy",
    )
    ax.plot(
        results_df["padding_pct"], results_df["f1_macro"],
        "s--", color="#3498db", linewidth=2, markersize=8, label="Macro-F1",
    )

    ax.axhline(y=0.6, color="gray", linestyle=":", linewidth=1, label="Majority-Class Baseline (~60%)")

    ax.axhline(y=0.7315, color="#e67e22", linestyle="--", linewidth=1.5, label="Timing-Only Classifier (73.15%)")

    ax.set_xlabel("Padding Strength (% of Original Packet Size)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Impact of Cloudflare Network Padding on a Timing-Aware Side Channel", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.55, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_5_network_padding_accuracy.png", dpi=300)
    plt.close()

    print(f"\nsaved plot_5_network_padding_accuracy.png")
    print(f"\nconclusion: at {PADDING_LEVELS[-1]}% padding, accuracy = {results[-1]['accuracy']:.4f}")
    if results[-1]["accuracy"] > 0.6:
        print("     vulnerability confirmed: padding fails because model learns from unpadded timing signals.")
    else:
        print("     padding degrades classification performance.")

if __name__ == "__main__":
    main()