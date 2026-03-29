#!/usr/bin/env python3
"""
data_collector.py — Primary data collection for LLM token-length side-channel research.

ARCHITECTURAL NOTE:
The primary data source is application-layer chunk data from the Groq Python library,
NOT raw TCP packets. Groq uses HTTPS/TLS, meaning TCP packet boundaries do not align
cleanly with token boundaries due to TLS framing and TCP segmentation. Chunk-level data
from the streaming API is the reliable per-token signal. Packet capture via scapy is
supplementary only — it provides network-level metadata but cannot directly reveal
individual token sizes.

RATE LIMITING NOTE:
The llama-3.1-8b-instant free tier has strict limits (30 RPM, 6,000 TPM, 500,000 TPD).
Collecting 1,000 samples WILL exceed the daily token limit, so this script is designed
to run across multiple days. The tenacity retry decorator handles HTTP 429 errors with
long exponential backoffs. The script appends to CSVs after every single API call and
checks for already-collected prompt_ids on startup, making it fully resumable — you can
kill and restart it at any time without losing progress or duplicating work.
"""

import csv
import json
import os
import socket
import subprocess
import sys
import time
import threading
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Scapy import — suppress noisy startup warnings
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import sniff, TCP, conf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMPTS_FILE = "prompts.csv"
CHUNK_DATA_FILE = "chunk_data.csv"
PACKET_DATA_FILE = "packet_data.csv"
FAILED_PROMPTS_FILE = "failed_prompts.csv"
INTER_PROMPT_DELAY = 3  # seconds between prompts
MODEL = "llama-3.1-8b-instant"


def load_env():
    """Load .env and return the Groq API key."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[FATAL] GROQ_API_KEY not found in environment. Create a .env file.")
        sys.exit(1)
    return api_key


def disable_hardware_offloading(iface_name: str):
    """
    Disables GRO, LRO, GSO, and TSO on the network interface.
    This prevents the Linux kernel/NIC from coalescing packets, ensuring 
    Scapy captures the true on-the-wire packet sizes (capped by MTU).
    """
    print(f"[*] Disabling hardware offloading on {iface_name}...")
    try:
        cmd = [
            "sudo", "ethtool", "-K", iface_name,
            "gro", "off", "lro", "off", "gso", "off", "tso", "off"
        ]
        # This will prompt for sudo password in the terminal if required
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK]  Hardware offloading disabled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to disable offloading on {iface_name}.")
        print(f"       Error: {e.stderr.strip()}")
        print("       Packet sizes may be artificially large (>1514 bytes).")
    except FileNotFoundError:
        print("[WARN] 'ethtool' not found. Please install it (e.g., sudo pacman -S ethtool).")
        print("       Packet sizes will likely be coalesced and artificially large.")


def verify_groq_connectivity(client: Groq) -> bool:
    """Send a single test request to verify API connectivity."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            stream=False,
        )
        print(f"[OK]  Groq API connectivity verified. Test response: {response.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"[FAIL] Groq API connectivity check failed: {e}")
        return False


def get_completed_ids() -> set:
    """
    Return set of prompt_ids already collected (for resume support).

    Reads chunk_data.csv and returns all prompt_ids that have been successfully
    saved. This allows the script to be killed and restarted without duplicating
    any work — it simply skips already-collected prompts.
    """
    completed = set()
    if os.path.exists(CHUNK_DATA_FILE):
        try:
            df = pd.read_csv(CHUNK_DATA_FILE)
            completed = set(df["prompt_id"].tolist())
        except Exception:
            pass
    return completed


def print_startup_summary(prompts_df: pd.DataFrame, skipped: int, iface: str):
    """Print a detailed startup summary."""
    total = len(prompts_df)
    remaining = total - skipped
    # Estimate: ~3s per prompt + backoff time is unpredictable
    eta_minutes = (remaining * INTER_PROMPT_DELAY) / 60
    # Daily token limit means we may only do ~500/day
    est_days = max(1, remaining // 500)

    print("\n" + "=" * 70)
    print("  LLM Token-Length Side-Channel — Data Collector")
    print("=" * 70)
    print(f"  Network interface (auto-detected): {iface}")
    print(f"  Model:                             {MODEL}")
    print(f"  Total prompts in dataset:          {total}")
    print(f"  Already collected (resuming):       {skipped}")
    print(f"  Remaining to collect:              {remaining}")
    print(f"  Min time (no rate limits):         ~{eta_minutes:.1f} minutes")
    print(f"  Estimated calendar time:           ~{est_days} day(s) (due to TPD limits)")
    print()

    # Breakdown by category and sensitivity
    print("  Category breakdown:")
    for sensitivity in ["High", "Low"]:
        subset = prompts_df[prompts_df["sensitivity"] == sensitivity]
        cats = subset["category"].value_counts().sort_index()
        for cat, count in cats.items():
            print(f"    [{sensitivity:4s}] {cat:<25s} {count} prompts")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Packet sniffer (supplementary — TLS-framed TCP segments, NOT raw token lengths)
# ---------------------------------------------------------------------------
class PacketSniffer:
    """
    Background packet sniffer capturing TCP port 443 traffic.

    NOTE: These are TLS-framed TCP segments, NOT individual token lengths.
    Due to TLS record framing and TCP segmentation, a single captured packet
    may contain multiple tokens or a partial token. This data is supplementary
    and is stored in a SEPARATE CSV — it is NOT merged row-by-row with chunk data.
    """

    def __init__(self, iface: str, groq_ip: str):
        self.iface = iface
        self.groq_ip = groq_ip
        self.packets = []
        self.timestamps = []
        self._stop_event = threading.Event()
        self._thread = None

    def _sniff_callback(self, pkt):
        if TCP in pkt and pkt[TCP].payload:
            payload_len = len(pkt[TCP].payload)
            self.packets.append(payload_len)
            self.timestamps.append(time.perf_counter())

    def _run_sniffer(self):
        bpf = f"tcp port 443 and host {self.groq_ip}" if self.groq_ip else "tcp port 443"
        try:
            sniff(
                iface=self.iface,
                filter=bpf,
                prn=self._sniff_callback,
                stop_filter=lambda _: self._stop_event.is_set(),
                store=False,
                timeout=120,  # safety timeout
            )
        except PermissionError:
            # Scapy needs CAP_NET_RAW — if unavailable, packet capture is silently skipped.
            # Chunk data (primary signal) is unaffected.
            pass

    def start(self):
        self.packets = []
        self.timestamps = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_sniffer, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.packets, self.timestamps


# ---------------------------------------------------------------------------
# Groq API call with tenacity retry for rate limits
# ---------------------------------------------------------------------------
@retry(
    # Retry on ANY exception — HTTP 429 from groq raises an exception,
    # as do transient network errors, timeouts, etc.
    retry=retry_if_exception_type(Exception),
    # Exponential backoff: 30s → 60s → 120s → 240s → 480s → ... up to 600s (10 min)
    # This is deliberately aggressive because the free tier has strict RPM/TPM/TPD limits.
    wait=wait_exponential(multiplier=2, min=30, max=600),
    # Up to 15 attempts — covers multi-minute rate-limit windows
    stop=stop_after_attempt(15),
    before_sleep=lambda retry_state: print(
        f"    [RATE LIMIT] Attempt {retry_state.attempt_number} hit rate limit. "
        f"Backing off {retry_state.next_action.sleep:.0f}s before retry..."
    ),
)
def stream_prompt_with_retry(client: Groq, prompt_text: str):
    """
    Stream a prompt through the Groq API with automatic retry on rate limits.

    The tenacity decorator catches HTTP 429 (rate limit exceeded) errors from the
    Groq client and applies exponential backoff. This is CRITICAL for the free tier:
      - 30 requests per minute (RPM)
      - 6,000 tokens per minute (TPM)
      - 500,000 tokens per day (TPD)

    With 1,000 prompts to collect, the daily limit will be hit. When that happens,
    the backoff grows to 10 minutes between retries, effectively pausing until the
    limit resets.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        stream=True,
    )
    return stream


# ---------------------------------------------------------------------------
# Core collection for a single prompt
# ---------------------------------------------------------------------------
def collect_prompt(client: Groq, prompt_text: str, sniffer: PacketSniffer):
    """
    Stream a single prompt through Groq and record chunk-level data (primary)
    and packet-level data (supplementary).

    Chunk data and packet data are kept as two ENTIRELY SEPARATE datasets.
    They share the same prompt_id for reference but are NOT merged row-by-row.

    Returns:
        chunk_sizes: list of byte sizes per streamed chunk
        chunk_timestamps: list of perf_counter timestamps per chunk
        packet_sizes: list of packet payload sizes
        packet_timestamps: list of packet timestamps
    """
    chunk_sizes = []
    chunk_timestamps = []

    # Start supplementary packet capture
    sniffer.start()
    time.sleep(0.2)  # brief settle time for sniffer thread

    # Stream the prompt via Groq API (with retry on rate limits)
    stream = stream_prompt_with_retry(client, prompt_text)

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            byte_size = len(content.encode("utf-8"))
            chunk_sizes.append(byte_size)
            chunk_timestamps.append(time.perf_counter())

    # Stop sniffer and retrieve packet data
    time.sleep(0.2)  # let trailing packets arrive
    packet_sizes, packet_timestamps = sniffer.stop()

    return chunk_sizes, chunk_timestamps, packet_sizes, packet_timestamps


def append_row(filepath: str, row: dict, fieldnames: list):
    """
    Append a single row to a CSV, creating the file with headers if needed.

    This is called after EVERY successful API call to ensure no data is lost
    if the script is interrupted. This is critical for multi-day collection runs.
    """
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_failure(prompt_id: int, category: str, sensitivity: str, error: str):
    """Log a failed prompt to failed_prompts.csv."""
    append_row(
        FAILED_PROMPTS_FILE,
        {
            "prompt_id": prompt_id,
            "category": category,
            "sensitivity": sensitivity,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
        },
        ["prompt_id", "category", "sensitivity", "timestamp", "error"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    api_key = load_env()
    client = Groq(api_key=api_key)

    # Verify connectivity
    if not verify_groq_connectivity(client):
        sys.exit(1)

    # Load prompts
    if not os.path.exists(PROMPTS_FILE):
        print(f"[FATAL] {PROMPTS_FILE} not found. Run generate_prompts.py first.")
        sys.exit(1)

    prompts_df = pd.read_csv(PROMPTS_FILE)
    total_prompts = len(prompts_df)

    # Resume support — check which prompt_ids are already in chunk_data.csv
    completed_ids = get_completed_ids()
    skipped = len(completed_ids)

    # Disable offloading on the active interface before starting
    iface_str = str(conf.iface)
    disable_hardware_offloading(iface_str)

    print_startup_summary(prompts_df, skipped, iface_str)

    if skipped == total_prompts:
        print("[INFO] All prompts already collected. Nothing to do.")
        return

    # Prepare sniffer — resolve Groq API IP to filter only relevant traffic
    try:
        groq_ip = socket.gethostbyname("api.groq.com")
        print(f"[OK]  Resolved api.groq.com → {groq_ip}")
    except socket.gaierror:
        groq_ip = None
        print("[WARN] Could not resolve api.groq.com — packet sniffer will capture all port 443 traffic")

    sniffer = PacketSniffer(iface_str, groq_ip) if groq_ip else PacketSniffer(iface_str, "")

    # CSV column definitions — chunk and packet data are SEPARATE files
    chunk_fields = [
        "prompt_id", "category", "sensitivity",
        "chunk_sizes", "timestamps", "total_chunks", "total_bytes",
    ]
    packet_fields = [
        "prompt_id", "category", "sensitivity",
        "packet_sizes", "packet_timestamps", "total_packets",
    ]

    # Collection loop
    total_collected = 0
    total_failed = 0
    start_time = time.time()
    category_chunks = {}
    category_bytes = {}

    for idx, row in prompts_df.iterrows():
        prompt_id = idx
        if prompt_id in completed_ids:
            continue

        prompt_text = row["prompt"]
        category = row["category"]
        sensitivity = row["sensitivity"]

        try:
            t0 = time.time()
            chunk_sizes, chunk_ts, packet_sizes, packet_ts = collect_prompt(
                client, prompt_text, sniffer
            )
            elapsed = time.time() - t0

            total_bytes = sum(chunk_sizes)
            total_chunks = len(chunk_sizes)

            # Save chunk data IMMEDIATELY after each API call (PRIMARY)
            # This ensures no data loss if the script is killed mid-run
            append_row(
                CHUNK_DATA_FILE,
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "sensitivity": sensitivity,
                    "chunk_sizes": json.dumps(chunk_sizes),
                    "timestamps": json.dumps(chunk_ts),
                    "total_chunks": total_chunks,
                    "total_bytes": total_bytes,
                },
                chunk_fields,
            )

            # Save packet data IMMEDIATELY (SUPPLEMENTARY — separate CSV, NOT merged)
            append_row(
                PACKET_DATA_FILE,
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "sensitivity": sensitivity,
                    "packet_sizes": json.dumps(packet_sizes),
                    "packet_timestamps": json.dumps(packet_ts),
                    "total_packets": len(packet_sizes),
                },
                packet_fields,
            )

            total_collected += 1

            # Track per-category stats
            category_chunks.setdefault(category, []).append(total_chunks)
            category_bytes.setdefault(category, []).append(total_bytes)

            print(
                f"  [{prompt_id + 1:4d}/{total_prompts}] "
                f"{category:<25s} [{sensitivity:4s}]  "
                f"chunks={total_chunks:4d}  bytes={total_bytes:6d}  "
                f"time={elapsed:.1f}s"
            )

        except Exception as e:
            # This only triggers AFTER tenacity exhausts all 15 retry attempts
            total_failed += 1
            log_failure(prompt_id, category, sensitivity, e)
            print(
                f"  [{prompt_id + 1:4d}/{total_prompts}] "
                f"{category:<25s} [{sensitivity:4s}]  "
                f"FAILED (after retries): {e}"
            )

        # Wait between prompts to respect rate limits
        time.sleep(INTER_PROMPT_DELAY)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  Collection Session Complete")
    print("=" * 70)
    print(f"  Collected this session: {total_collected}")
    print(f"  Failed this session:    {total_failed}")
    print(f"  Previously collected:   {skipped}")
    print(f"  Total in chunk_data:    {skipped + total_collected}")
    print(f"  Session time:           {total_time / 60:.1f} minutes")
    print()

    if category_chunks:
        print("  Average chunks per category (this session):")
        for cat in sorted(category_chunks.keys()):
            avg_c = sum(category_chunks[cat]) / len(category_chunks[cat])
            avg_b = sum(category_bytes[cat]) / len(category_bytes[cat])
            print(f"    {cat:<25s} avg_chunks={avg_c:.1f}  avg_bytes={avg_b:.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()