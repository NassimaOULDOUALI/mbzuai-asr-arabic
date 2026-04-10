"""
data.py
Dataset loading and preprocessing for Arabic ASR fine-tuning.
Dataset: Mozilla Common Voice 17.0 - Arabic (ar)
"""

import os
import re
import numpy as np
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG — change only here
# ─────────────────────────────────────────────────────────────
DATASET_NAME     = "mozilla-foundation/common_voice_17_0"
LANGUAGE         = "ar"
MODEL_NAME       = "openai/whisper-small"
SAMPLING_RATE    = 16_000
MIN_DURATION_S   = 1.0
MAX_DURATION_S   = 15.0
MAX_TRAIN_HOURS  = 8.0          # cap training set at ~8h
SEED             = 42

# ─────────────────────────────────────────────────────────────
# TEXT NORMALIZATION FOR ARABIC
# ─────────────────────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic transcription text.
    - Remove Latin characters and punctuation
    - Keep Arabic script and diacritics (tashkeel)
    - Normalize whitespace
    """
    # Remove Latin characters
    text = re.sub(r'[a-zA-Z]', '', text)
    # Remove punctuation except Arabic-specific chars
    text = re.sub(r'[،؟!؛\.\,\?\!\;\:\-\(\)\[\]]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# DURATION FILTER
# ─────────────────────────────────────────────────────────────
def is_valid_duration(example):
    """Keep only samples between MIN and MAX duration."""
    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return MIN_DURATION_S <= duration <= MAX_DURATION_S


def is_valid_text(example):
    """Remove samples with empty transcriptions."""
    return len(example["sentence"].strip()) > 0


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def prepare_dataset(example, feature_extractor, tokenizer):
    """
    Convert raw audio + text into Whisper model inputs.
    - Compute log-mel spectrogram from audio
    - Tokenize transcription
    """
    audio = example["audio"]

    # Compute log-mel spectrogram
    example["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Normalize and tokenize transcription
    sentence = normalize_arabic(example["sentence"])
    example["labels"] = tokenizer(sentence).input_ids

    return example


# ─────────────────────────────────────────────────────────────
# DATASET STATISTICS
# ─────────────────────────────────────────────────────────────
def print_dataset_stats(dataset: DatasetDict):
    """Print key statistics about the dataset splits."""
    print("\n" + "="*50)
    print("DATASET STATISTICS — Common Voice 17.0 Arabic")
    print("="*50)

    for split_name, split in dataset.items():
        durations = [
            len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            for ex in tqdm(split, desc=f"Computing {split_name} stats")
        ]
        total_hours = sum(durations) / 3600
        avg_duration = np.mean(durations)
        num_speakers = len(set(split["client_id"])) if "client_id" in split.column_names else "N/A"

        print(f"\n[{split_name.upper()}]")
        print(f"  Samples       : {len(split)}")
        print(f"  Total duration: {total_hours:.2f}h")
        print(f"  Avg duration  : {avg_duration:.2f}s")
        print(f"  Speakers      : {num_speakers}")

    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN LOADING FUNCTION
# ─────────────────────────────────────────────────────────────
def load_and_prepare_data(max_train_hours=MAX_TRAIN_HOURS):
    """
    Full pipeline: load → filter → cap → extract features.
    Returns processed DatasetDict ready for training.
    """

    print("Loading Common Voice 17.0 Arabic...")
    raw = DatasetDict({
        "train": load_dataset(
            DATASET_NAME, LANGUAGE,
            split="train",
            trust_remote_code=True,
            token=True
        ),
        "validation": load_dataset(
            DATASET_NAME, LANGUAGE,
            split="validation",
            trust_remote_code=True,
            token=True
        ),
        "test": load_dataset(
            DATASET_NAME, LANGUAGE,
            split="test",
            trust_remote_code=True,
            token=True
        ),
    })

    # Resample to 16kHz
    print("Resampling to 16kHz...")
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Filter: valid text + valid duration
    print("Filtering samples...")
    raw = raw.filter(is_valid_text)
    raw = raw.filter(is_valid_duration)

    # Cap training set
    max_train_samples = int(max_train_hours * 3600 / 5)  # ~5s avg duration
    if len(raw["train"]) > max_train_samples:
        raw["train"] = raw["train"].shuffle(seed=SEED).select(
            range(max_train_samples)
        )
        print(f"Training set capped at {max_train_samples} samples (~{max_train_hours}h)")

    # Print statistics BEFORE feature extraction
    print_dataset_stats(raw)

    # Load feature extractor and tokenizer
    print("Loading Whisper feature extractor and tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME,
        language="arabic",
        task="transcribe"
    )

    # Remove unused columns
    columns_to_remove = [
        col for col in raw["train"].column_names
        if col not in ["audio", "sentence"]
    ]

    # Apply feature extraction
    print("Extracting features...")
    processed = raw.map(
        lambda ex: prepare_dataset(ex, feature_extractor, tokenizer),
        remove_columns=columns_to_remove,
        num_proc=4,
        desc="Preprocessing dataset"
    )

    return processed, feature_extractor, tokenizer


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset, fe, tok = load_and_prepare_data(max_train_hours=2.0)
    print("Sample keys:", dataset["train"][0].keys())
    print("Input features shape:", dataset["train"][0]["input_features"].shape)
    print("Labels sample:", dataset["train"][0]["labels"][:10])
    print("Done.")
