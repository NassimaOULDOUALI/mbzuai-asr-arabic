"""
01_download.py
Download Common Voice 17.0 Arabic and save to disk.
Run once — then work offline from local files.
"""

import os
from datasets import load_dataset, DatasetDict, Audio

# ── CONFIG ────────────────────────────────────────────────────
DATASET_NAME   = "mozilla-foundation/common_voice_17_0"
LANGUAGE       = "ar"
SAMPLING_RATE  = 16_000
SAVE_DIR       = "./data/raw"
# ─────────────────────────────────────────────────────────────

def download():
    print(f"Downloading {DATASET_NAME} [{LANGUAGE}]...")
    print("This may take a few minutes — audio files are large.\n")

    dataset = DatasetDict({
        split: load_dataset(
            DATASET_NAME,
            LANGUAGE,
            split=split,
            token=True         # utilise ton token HF
        )
        for split in ["train", "validation", "test"]
    })

    # Resample tout à 16kHz dès maintenant
    print("Resampling to 16kHz...")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE)
    )

    # Sauvegarde locale
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving to {SAVE_DIR}...")
    dataset.save_to_disk(SAVE_DIR)

    print("\nDone. Files saved to:", SAVE_DIR)
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)} samples")


if __name__ == "__main__":
    download()
