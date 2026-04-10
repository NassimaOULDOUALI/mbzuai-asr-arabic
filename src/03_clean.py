"""
03_clean.py
Clean and filter the raw dataset.
Filters:
  - Valid duration (1s - 15s)
  - Non-empty transcription
  - Remove Latin characters from transcription
  - Cap train split at MAX_TRAIN_HOURS
Run after 02_stats.py.
"""

import re
import os
import numpy as np
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────
RAW_DIR          = "./data/raw"
CLEAN_DIR        = "./data/clean"
MIN_DURATION_S   = 1.0
MAX_DURATION_S   = 15.0
MAX_TRAIN_HOURS  = 8.0
SEED             = 42
# ─────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text:
    - Remove Latin characters
    - Remove punctuation
    - Keep Arabic script and diacritics (tashkeel)
    - Normalize whitespace
    """
    # Remove Latin
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    # Remove punctuation (Arabic + Latin)
    text = re.sub(r'[،؟!؛\.\,\?\!\;\:\-\(\)\[\]\"\']+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_duration(example) -> float:
    arr = example["audio"]["array"]
    sr  = example["audio"]["sampling_rate"]
    return len(arr) / sr


def filter_duration(example) -> bool:
    d = get_duration(example)
    return MIN_DURATION_S <= d <= MAX_DURATION_S


def filter_text(example) -> bool:
    cleaned = normalize_arabic(example["sentence"])
    return len(cleaned) > 0


def apply_normalization(example):
    example["sentence"] = normalize_arabic(example["sentence"])
    return example


def cap_train_split(train_split, max_hours: float):
    """Cap training set to max_hours."""
    durations = np.array([
        get_duration(ex)
        for ex in tqdm(train_split, desc="Computing durations for cap")
    ])
    cumsum  = np.cumsum(durations)
    max_sec = max_hours * 3600
    cutoff  = int(np.searchsorted(cumsum, max_sec))

    print(f"  Capping train: {len(train_split)} → {cutoff} samples "
          f"({cumsum[cutoff-1]/3600:.2f}h)")

    return train_split.shuffle(seed=SEED).select(range(cutoff))


def main():
    print("Loading raw dataset...")
    dataset = load_from_disk(RAW_DIR)

    print("\nBefore cleaning:")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} samples")

    # Step 1 — Filter duration
    print("\nFiltering by duration...")
    dataset = dataset.filter(
        filter_duration,
        desc="Duration filter"
    )

    # Step 2 — Filter empty transcriptions
    print("Filtering empty transcriptions...")
    dataset = dataset.filter(
        filter_text,
        desc="Text filter"
    )

    # Step 3 — Normalize Arabic text
    print("Normalizing Arabic text...")
    dataset = dataset.map(
        apply_normalization,
        desc="Text normalization"
    )

    # Step 4 — Cap training set
    print(f"\nCapping train to {MAX_TRAIN_HOURS}h...")
    dataset["train"] = cap_train_split(
        dataset["train"], MAX_TRAIN_HOURS
    )

    print("\nAfter cleaning:")
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} samples")

    # Save
    os.makedirs(CLEAN_DIR, exist_ok=True)
    print(f"\nSaving clean dataset to {CLEAN_DIR}...")
    dataset.save_to_disk(CLEAN_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
