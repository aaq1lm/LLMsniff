#!/usr/bin/env python3
"""
feature_extractor.py — Extract statistical features from chunk and packet data.

ARCHITECTURAL NOTE:
Chunk-level features establish the theoretical baseline (Oracle). 
Packet-level features simulate the real-world passive network observer.
Timing features (ICT/IPT) are extracted to capture LLM "think time" variations.
"""

import json
import sys
import numpy as np
import pandas as pd

CHUNK_DATA_FILE = "chunk_data.csv"
PACKET_DATA_FILE = "packet_data.csv"
FEATURES_FILE = "features.csv"

# Number of sequence columns to generate for both chunks and packets
SEQUENCE_LENGTH = 250

def extract_chunk_features(row: pd.Series) -> dict:
    """Extract primary chunk-based features and timing."""
    sizes = np.array(json.loads(row["chunk_sizes"]), dtype=float)
    timestamps = np.array(json.loads(row["timestamps"]), dtype=float)
    
    # Calculate Inter-Chunk Arrival Times (ICT)
    ict = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])

    features = {
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "sensitivity": row["sensitivity"],
        
        # Size Stats
        "chunk_mean": np.mean(sizes) if len(sizes) > 0 else 0.0,
        "chunk_std": np.std(sizes) if len(sizes) > 0 else 0.0,
        "chunk_max": np.max(sizes) if len(sizes) > 0 else 0.0,
        "chunk_min": np.min(sizes) if len(sizes) > 0 else 0.0,
        "chunk_median": np.median(sizes) if len(sizes) > 0 else 0.0,
        
        # Timing Stats
        "chunk_ict_mean": np.mean(ict) if len(ict) > 0 else 0.0,
        "chunk_ict_std": np.std(ict) if len(ict) > 0 else 0.0,
        "chunk_ict_max": np.max(ict) if len(ict) > 0 else 0.0,
        
        "total_chunks": int(row["total_chunks"]),
        "total_bytes": int(row["total_bytes"]),
    }

    # Fixed-length chunk size sequence
    padded = np.zeros(SEQUENCE_LENGTH, dtype=float)
    n = min(len(sizes), SEQUENCE_LENGTH)
    if n > 0:
        padded[:n] = sizes[:n]
    for i in range(SEQUENCE_LENGTH):
        features[f"chunk_seq_{i}"] = padded[i]

    return features

def extract_packet_features(row: pd.Series) -> dict:
    """Extract supplementary packet-based features and timing."""
    sizes = np.array(json.loads(row["packet_sizes"]), dtype=float)
    timestamps = np.array(json.loads(row["packet_timestamps"]), dtype=float)
    
    # Calculate Inter-Packet Arrival Times (IPT)
    ipt = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])

    features = {
        # Size Stats
        "packet_mean": np.mean(sizes) if len(sizes) > 0 else 0.0,
        "packet_std": np.std(sizes) if len(sizes) > 0 else 0.0,
        "packet_max": np.max(sizes) if len(sizes) > 0 else 0.0,
        "packet_min": np.min(sizes) if len(sizes) > 0 else 0.0,
        "packet_median": np.median(sizes) if len(sizes) > 0 else 0.0,
        
        # Timing Stats
        "packet_ipt_mean": np.mean(ipt) if len(ipt) > 0 else 0.0,
        "packet_ipt_std": np.std(ipt) if len(ipt) > 0 else 0.0,
        "packet_ipt_max": np.max(ipt) if len(ipt) > 0 else 0.0,
        
        "total_packets": int(row["total_packets"]) if pd.notna(row["total_packets"]) else 0,
    }

    # Fixed-length packet size sequence
    padded = np.zeros(SEQUENCE_LENGTH, dtype=float)
    n = min(len(sizes), SEQUENCE_LENGTH)
    if n > 0:
        padded[:n] = sizes[:n]
    for i in range(SEQUENCE_LENGTH):
        features[f"packet_seq_{i}"] = padded[i]

    return features

def main():
    try:
        chunk_df = pd.read_csv(CHUNK_DATA_FILE)
    except FileNotFoundError:
        print(f"[FATAL] {CHUNK_DATA_FILE} not found. Run data_collector.py first.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(chunk_df)} rows from {CHUNK_DATA_FILE}")

    packet_df = None
    try:
        packet_df = pd.read_csv(PACKET_DATA_FILE)
        print(f"[INFO] Loaded {len(packet_df)} rows from {PACKET_DATA_FILE}")
    except FileNotFoundError:
        print(f"[WARN] {PACKET_DATA_FILE} not found. Packet features will be zero.")

    all_features = []

    for _, row in chunk_df.iterrows():
        # Get primary features
        features = extract_chunk_features(row)

        # Get supplementary features (if available)
        if packet_df is not None:
            pkt_row = packet_df[packet_df["prompt_id"] == row["prompt_id"]]
            if not pkt_row.empty:
                pkt_features = extract_packet_features(pkt_row.iloc[0])
                features.update(pkt_features) 
            else:
                features.update(extract_packet_features(pd.Series({"packet_sizes": "[]", "packet_timestamps": "[]", "total_packets": 0})))
        else:
            features.update(extract_packet_features(pd.Series({"packet_sizes": "[]", "packet_timestamps": "[]", "total_packets": 0})))

        all_features.append(features)

    # Save to CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(FEATURES_FILE, index=False)

    n_cols = len(features_df.columns)
    print(f"\n[OK] Saved {len(features_df)} rows × {n_cols} columns to {FEATURES_FILE}")
    print("     Matrix now includes full sequence padding and IPT/ICT timing features.")

if __name__ == "__main__":
    main()