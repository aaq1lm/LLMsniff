#!/usr/bin/env python3
"""
real_world_framing.py — Generate a plain-English privacy impact summary from
the classification results.

Reads results_summary.csv and prints a publication-ready paragraph that
contextualizes the findings for a general audience and research paper
results section.
"""

import sys

import pandas as pd


RESULTS_FILE = "results_summary.csv"


def main():
    try:
        df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"[FATAL] {RESULTS_FILE} not found. Run classifier.py first.")
        sys.exit(1)

    # ---- Extract key metrics ----

    # FIX: Use explicit "Attacker" filter so we don't accidentally grab the Oracle row.
    # The original .iloc[0] on a "Binary" match returned "Oracle (Chunk) - Binary Sensitivity"
    # because it sorts first alphabetically, reporting oracle numbers as attacker numbers.

    # Attacker (Packet) — Binary sensitivity classifier
    binary_row = df[df["classifier"].str.contains("Attacker.*Binary", case=False)].iloc[0]
    binary_acc = binary_row["mean_accuracy"] * 100
    binary_f1 = binary_row["mean_macro_f1"] * 100

    # Attacker (Packet) — 10-class domain classifier
    domain_row = df[df["classifier"].str.contains("Attacker.*10-Class", case=False)].iloc[0]
    domain_acc = domain_row["mean_accuracy"] * 100

    # Attacker (Packet) — Sensitive-only 6-class classifier
    sensitive_row = df[df["classifier"].str.contains("Attacker.*Sensitive", case=False)].iloc[0]
    sensitive_acc = sensitive_row["mean_accuracy"] * 100

    # Also extract Oracle numbers for comparison
    oracle_binary_row = df[df["classifier"].str.contains("Oracle.*Binary", case=False)].iloc[0]
    oracle_binary_acc = oracle_binary_row["mean_accuracy"] * 100

    # Per-class F1 scores for sensitive categories
    sensitive_f1_cols = [c for c in sensitive_row.index if c.startswith("f1_")]
    sensitive_f1 = {}
    for col in sensitive_f1_cols:
        cat_name = col.replace("f1_", "")
        val = sensitive_row[col]
        if pd.notna(val):
            sensitive_f1[cat_name] = val * 100

    # Find most and least identifiable sensitive category
    if sensitive_f1:
        most_identifiable = max(sensitive_f1, key=sensitive_f1.get)
        least_identifiable = min(sensitive_f1, key=sensitive_f1.get)
        most_f1 = sensitive_f1[most_identifiable]
        least_f1 = sensitive_f1[least_identifiable]
    else:
        most_identifiable = "N/A"
        least_identifiable = "N/A"
        most_f1 = 0
        least_f1 = 0

    # Per-class F1 for Mental_Health from the binary or domain classifier
    mental_health_f1 = sensitive_f1.get("Mental_Health", 0)
    # Convert F1 to approximate "X out of 10" framing
    mental_health_out_of_10 = round(mental_health_f1 / 10)

    # ---- Format the human-readable name ----
    def humanize(cat: str) -> str:
        return cat.replace("_", " ").lower()

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("  Privacy Impact Summary — LLM Token-Length Side-Channel Attack")
    print("=" * 70)

    paragraph = (
        f"Our experiment demonstrates that a passive network observer — one with "
        f"no ability to decrypt TLS traffic — could correctly classify whether an "
        f"LLM interaction involves a sensitive topic (mental health, medical symptoms, "
        f"legal trouble, financial distress, substance use, or personal crisis) with "
        f"{binary_acc:.1f}% accuracy and {binary_f1:.1f}% macro-F1 score, using only "
        f"the byte sizes and timing of streamed response packets observable through "
        f"network metadata. For comparison, the theoretical Oracle — which has direct "
        f"access to raw application-layer token sizes before TLS encryption — achieves "
        f"{oracle_binary_acc:.1f}% accuracy, placing our attacker at "
        f"{binary_acc / oracle_binary_acc * 100:.0f}% of the theoretical ceiling. "
        f"Beyond binary sensitivity detection, the attacker could identify the specific "
        f"topic domain among all 10 categories with {domain_acc:.1f}% accuracy (vs. "
        f"10% random chance). Even when restricting analysis to sensitive topics only, "
        f"the attacker could distinguish between the six sensitive categories with "
        f"{sensitive_acc:.1f}% accuracy. "
        f"Mental health queries were correctly identified approximately "
        f"{mental_health_out_of_10} out of 10 times (F1 = {mental_health_f1:.1f}%). "
        f"Among sensitive topics, {humanize(most_identifiable)} queries were the most "
        f"identifiable at {most_f1:.1f}% F1 score, while {humanize(least_identifiable)} "
        f"queries were the least identifiable at {least_f1:.1f}% F1 score. "
        f"These results were obtained using 5-fold stratified cross-validation on "
        f"a dataset of 2,000 prompts (200 per category) collected over the Groq "
        f"streaming API, with all traffic encrypted under TLS 1.3. The classification "
        f"relies solely on packet-level byte sizes and inter-packet timing metadata — "
        f"no content, headers, or decryption keys were used."
    )

    # Word-wrap for readability
    import textwrap
    wrapped = textwrap.fill(paragraph, width=78)
    print()
    print(wrapped)
    print()
    print("=" * 70)

    # Also print individual stats for easy reference
    print("\n  Quick Reference:")
    print(f"    Binary (High vs Low):    {binary_acc:.1f}% accuracy")
    print(f"    10-class domain:         {domain_acc:.1f}% accuracy")
    print(f"    Sensitive 6-class:       {sensitive_acc:.1f}% accuracy")
    if sensitive_f1:
        print(f"\n  Per-category F1 (sensitive only):")
        for cat, f1 in sorted(sensitive_f1.items(), key=lambda x: -x[1]):
            print(f"    {cat:<25s} {f1:.1f}%")


if __name__ == "__main__":
    main()
