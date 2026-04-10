"""
02_stats.py
Compute and display statistics on raw downloaded data.
Run after 01_download.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────
SAVE_DIR    = "./data/raw"
FIGURES_DIR = "./results/figures"
# ─────────────────────────────────────────────────────────────

def compute_stats(split_name, split):
    """Compute duration, speaker and text statistics."""
    durations = []
    for ex in tqdm(split, desc=f"  {split_name}"):
        arr = ex["audio"]["array"]
        sr  = ex["audio"]["sampling_rate"]
        durations.append(len(arr) / sr)

    durations = np.array(durations)
    texts     = split["sentence"]

    stats = {
        "split"          : split_name,
        "num_samples"    : len(split),
        "total_hours"    : durations.sum() / 3600,
        "avg_duration_s" : durations.mean(),
        "min_duration_s" : durations.min(),
        "max_duration_s" : durations.max(),
        "num_speakers"   : len(set(split["client_id"])),
        "avg_text_len"   : np.mean([len(t) for t in texts]),
        "vocab_size"     : len(set(" ".join(texts).split())),
    }
    return stats, durations


def plot_duration_distribution(durations_dict):
    """Plot duration histogram per split."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (split_name, durations) in zip(axes, durations_dict.items()):
        ax.hist(durations, bins=50, color="#003A70", edgecolor="white")
        ax.set_title(f"{split_name} — duration distribution")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.axvline(durations.mean(), color="red",
                   linestyle="--", label=f"mean={durations.mean():.1f}s")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "duration_distribution.png")
    plt.savefig(path, dpi=150)
    print(f"Figure saved: {path}")
    plt.show()


def main():
    import os
    print("Loading dataset from disk...")
    dataset = load_from_disk(SAVE_DIR)

    all_stats    = []
    all_durations = {}

    for split_name, split in dataset.items():
        print(f"\nComputing stats for [{split_name}]...")
        stats, durations = compute_stats(split_name, split)
        all_stats.append(stats)
        all_durations[split_name] = durations

    # Print table
    df = pd.DataFrame(all_stats).set_index("split")
    print("\n" + "="*60)
    print("DATASET STATISTICS — Common Voice 17.0 Arabic")
    print("="*60)
    print(df.to_string())
    print("="*60)

    # Save stats
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/dataset_stats.csv")
    print("\nStats saved to ./results/dataset_stats.csv")

    # Plot
    plot_duration_distribution(all_durations)


if __name__ == "__main__":
    main()
