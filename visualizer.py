#!/usr/bin/env python3
"""
visualizer.py — Generate publication-quality plots for the LLM token-length
side-channel attack research paper.

Figures generated:
  plot_2_feature_importance.png  — Figure 1: Top 20 feature importances (Oracle + Attacker)
  plot_3_per_category_f1.png     — Figure 2: Per-category F1 Oracle vs Attacker (10-class)
  plot_3_sequence_rhythm.png     — Supplementary: High vs Low sequence rhythm
  plot_4_response_length.png     — Supplementary: Average response length per category

Figure 3 (confusion matrix): confusion_matrix_10class_packet.png (from classifier.py)
Figure 4 (padding curve):    plot_5_network_padding_accuracy.png (from mitigation_simulator.py)
"""

import sys
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

CHUNK_DATA_FILE = "chunk_data.csv"
FEATURES_FILE   = "features.csv"
RESULTS_FILE    = "results_summary.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

SENSITIVE_CATEGORIES = {
    "Mental_Health", "Medical_Symptoms", "Legal_Trouble",
    "Financial_Distress", "Substance_Use", "Personal_Crisis",
}


def load_data():
    try:
        chunk_df = pd.read_csv(CHUNK_DATA_FILE)
    except FileNotFoundError:
        print(f"[FATAL] {CHUNK_DATA_FILE} not found.")
        sys.exit(1)
    try:
        features_df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"[FATAL] {FEATURES_FILE} not found.")
        sys.exit(1)
    return chunk_df, features_df


def explode_chunk_sizes(chunk_df):
    rows = []
    for _, row in chunk_df.iterrows():
        for s in json.loads(row["chunk_sizes"]):
            rows.append({
                "category": row["category"],
                "sensitivity": row["sensitivity"],
                "chunk_size": s
            })
    return pd.DataFrame(rows)


def plot_feature_importance():
    """Figure 1: Top 20 features side by side for Oracle and Attacker binary classifiers."""
    oracle_file   = "feature_importance_Oracle_Chunk_-_Binary_Sensitivity.csv"
    attacker_file = "feature_importance_Attacker_Packet_-_Binary_Sensitivity.csv"

    missing = [f for f in [oracle_file, attacker_file] if not os.path.exists(f)]
    if missing:
        print(f"[SKIP] Feature importance CSVs not found: {missing}")
        print("       Re-run classifier.py first to generate them.")
        return

    oracle_df   = pd.read_csv(oracle_file).head(20)
    attacker_df = pd.read_csv(attacker_file).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, title, color in zip(
        axes,
        [oracle_df, attacker_df],
        ["(a) Oracle — Top 20 Features", "(b) Attacker — Top 20 Features"],
        ["#2ecc71", "#e74c3c"]
    ):
        ax.barh(df["feature"][::-1], df["importance"][::-1], color=color, edgecolor="white")
        ax.set_xlabel("Mean Feature Importance (across 5 folds)")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=9)

    plt.suptitle(
        "Random Forest Feature Importance — Binary Sensitivity Classifier",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("plot_2_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved plot_2_feature_importance.png")


def plot_per_category_f1():
    """Figure 2: Grouped bar chart of per-category F1 for Oracle vs Attacker 10-class."""
    try:
        results_df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"[SKIP] {RESULTS_FILE} not found. Run classifier.py first.")
        return

    oracle_row   = results_df[results_df["classifier"] == "Oracle (Chunk) - 10-Class Domain"].iloc[0]
    attacker_row = results_df[results_df["classifier"] == "Attacker (Packet) - 10-Class Domain"].iloc[0]

    categories = [
        "Coding_Help", "Cooking_Recipes", "Financial_Distress", "General_Knowledge",
        "Legal_Trouble", "Medical_Symptoms", "Mental_Health",
        "Personal_Crisis", "Substance_Use", "Travel_Planning"
    ]
    clean_labels = [c.replace("_", " ") for c in categories]
    oracle_f1    = [oracle_row.get(f"f1_{c}", 0) for c in categories]
    attacker_f1  = [attacker_row.get(f"f1_{c}", 0) for c in categories]

    x     = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, oracle_f1,   width, label="Oracle (Chunk)",    color="#2ecc71", edgecolor="white")
    ax.bar(x + width/2, attacker_f1, width, label="Attacker (Packet)", color="#e74c3c", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=45, ha="right")
    for tick, cat in zip(ax.get_xticklabels(), categories):
        if cat in SENSITIVE_CATEGORIES:
            tick.set_color("#c0392b")

    ax.axhline(y=0.10, color="gray", linestyle=":", linewidth=1, label="Random chance (10%)")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Per-Category F1 Scores — Oracle vs. Attacker (10-Class Domain Classifier)")
    ax.legend()
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        "Red labels = Sensitive categories",
        xy=(0.01, 0.97), xycoords="axes fraction",
        fontsize=9, color="#c0392b", va="top"
    )

    plt.tight_layout()
    plt.savefig("plot_3_per_category_f1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved plot_3_per_category_f1.png")


def plot_sequence_rhythm(features_df):
    """Supplementary: Average chunk sequence shape for High vs Low sensitivity."""
    seq_cols  = [f"chunk_seq_{i}" for i in range(100)]
    high_mean = features_df[features_df["sensitivity"] == "High"][seq_cols].mean().values
    low_mean  = features_df[features_df["sensitivity"] == "Low"][seq_cols].mean().values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(high_mean, label="Sensitive (High)",    color="#e74c3c", linewidth=2)
    ax.plot(low_mean,  label="Non-Sensitive (Low)", color="#3498db", linewidth=2, alpha=0.8)
    ax.set_xlabel("Token Sequence Index (0–100)")
    ax.set_ylabel("Average Chunk Size (bytes)")
    ax.set_title("Side-Channel Rhythm: Structural Patterns in LLM Responses")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plot_3_sequence_rhythm.png", dpi=300)
    plt.close()
    print("[OK] Saved plot_3_sequence_rhythm.png")


def plot_response_length(features_df):
    """Supplementary: Average response length per category."""
    stats = (
        features_df.groupby(["category", "sensitivity"])["total_bytes"]
        .mean().reset_index().sort_values("total_bytes")
    )
    colors = ["#e74c3c" if s == "High" else "#3498db" for s in stats["sensitivity"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(range(len(stats)), stats["total_bytes"], color=colors, edgecolor="white")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["category"], rotation=45, ha="right")
    ax.set_ylabel("Average Total Response Size (bytes)")
    ax.set_title("Average Response Length by Category")
    ax.legend(handles=[
        Patch(facecolor="#e74c3c", label="Sensitive (High)"),
        Patch(facecolor="#3498db", label="Non-Sensitive (Low)")
    ], loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_4_response_length.png", dpi=300)
    plt.close()
    print("[OK] Saved plot_4_response_length.png")


def main():
    chunk_df, features_df = load_data()
    print(f"[INFO] Loaded {len(chunk_df)} rows from {CHUNK_DATA_FILE}")
    print(f"[INFO] Loaded {len(features_df)} rows from {FEATURES_FILE}\n")

    plot_feature_importance()
    plot_per_category_f1()
    plot_sequence_rhythm(features_df)
    plot_response_length(features_df)

    print("\n[OK] All plots saved at 300 dpi")
    print("     Figure 3 (confusion matrix): confusion_matrix_10class_packet.png")
    print("     Figure 4 (padding curve):    plot_5_network_padding_accuracy.png")


if __name__ == "__main__":
    main()