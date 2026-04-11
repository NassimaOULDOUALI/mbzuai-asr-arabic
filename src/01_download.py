"""
01_download.py
Download Common Voice 18 Arabic — full dataset, raw save.
Preprocessing is handled separately in 02_preprocess.py
"""

import os
from datasets import load_dataset, DatasetDict, Audio

# ── CONFIG ────────────────────────────────────────────────────
DATASET_NAME  = "MohamedRashad/common-voice-18-arabic"
SAMPLING_RATE = 16_000
SAVE_DIR      = "./data/raw"
# ─────────────────────────────────────────────────────────────

def download():
    print(f"Downloading {DATASET_NAME}...")
    print("Downloading all splits — preprocessing handled separately\n")

    dataset = DatasetDict({
        split: load_dataset(
            DATASET_NAME,
            split=split,
        )
        for split in ["train", "validation", "test"]
    })

    print(f"Resampling 48kHz → 16kHz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving raw dataset to {SAVE_DIR}...")
    dataset.save_to_disk(SAVE_DIR)

    print("\n✓ Done. Raw files saved to:", SAVE_DIR)
    for split, ds in dataset.items():
        print(f"  {split:12s}: {len(ds):>6,} samples")

    print("\nNext step: run 02_preprocess.py")


if __name__ == "__main__":
    download()
