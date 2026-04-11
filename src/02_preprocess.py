"""
02_preprocess.py
"""

import os
import re
import random
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

from datasets import load_from_disk, DatasetDict, Dataset, Audio, Features
import pyarrow as pa

# ── CONFIG ────────────────────────────────────────────────────
RAW_DIR        = "./data/raw"
PROCESSED_DIR  = "./data/processed"
TRAIN_SAMPLES  = 5_000
MIN_UPVOTES    = 2
SEED           = 42
# ─────────────────────────────────────────────────────────────

KEEP_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]"
)

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"[ىئ]",   "ي", text)
    text = re.sub(r"ة",       "ه", text)
    text = re.sub(r"ؤ",       "و", text)
    text = KEEP_ARABIC.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_split(name: str, ds):
    print(f"\n── {name.upper()} {'─'*40}")
    n0 = len(ds)

    # Lit toutes les colonnes NON-audio directement
    up_votes  = ds["up_votes"]
    sentences = ds["sentence"]

    # Colonnes métadonnées à conserver
    client_ids  = ds["client_id"]
    paths       = ds["path"]
    down_votes  = ds["down_votes"]
    ages        = ds["age"]
    genders     = ds["gender"]
    accents     = ds["accent"]
    locales     = ds["locale"]
    segments    = ds["segment"]
    variants    = ds["variant"]

    # 1. Filtre + normalisation en Python pur
    kept_indices   = []
    norm_sentences = []

    for i in range(n0):
        if up_votes[i] < MIN_UPVOTES:
            continue
        s = sentences[i]
        if not s or len(s.strip()) < 2:
            continue
        s_norm = normalize_arabic(s)
        if len(s_norm) < 2:
            continue
        kept_indices.append(i)
        norm_sentences.append(s_norm)

    print(f"  After filter + clean : {len(kept_indices):>6,} / {n0:>6,} kept")

    # 2. Sous-échantillonnage train
    if name == "train" and len(kept_indices) > TRAIN_SAMPLES:
        random.seed(SEED)
        sampled = sorted(random.sample(range(len(kept_indices)), TRAIN_SAMPLES))
        kept_indices   = [kept_indices[i]   for i in sampled]
        norm_sentences = [norm_sentences[i] for i in sampled]
        print(f"  After subsample      : {len(kept_indices):>6,} samples (~{TRAIN_SAMPLES*4.5/3600:.2f}h)")

    # 3. Reconstruit le dict sans jamais décoder l'audio
    # L'audio est un struct Arrow {bytes, path} — on le lit comme liste brute
    raw_table = ds._data.table
    audio_col = raw_table.column("audio").take(
        pa.array(kept_indices)
    ).to_pylist()  # liste de dicts {"bytes": b"...", "path": "..."}

    result = {
        "client_id"  : [client_ids[i]  for i in kept_indices],
        "path"       : [paths[i]       for i in kept_indices],
        "audio"      : audio_col,
        "sentence"   : norm_sentences,
        "up_votes"   : [up_votes[i]    for i in kept_indices],
        "down_votes" : [down_votes[i]  for i in kept_indices],
        "age"        : [ages[i]        for i in kept_indices],
        "gender"     : [genders[i]     for i in kept_indices],
        "accent"     : [accents[i]     for i in kept_indices],
        "locale"     : [locales[i]     for i in kept_indices],
        "segment"    : [segments[i]    for i in kept_indices],
        "variant"    : [variants[i]    for i in kept_indices],
    }

    ds_out = Dataset.from_dict(result, features=ds.features)
    return ds_out


def main():
    print(f"Loading raw dataset from {RAW_DIR}...")
    raw = load_from_disk(RAW_DIR)

    processed = DatasetDict({
        split: preprocess_split(split, raw[split])
        for split in ["train", "validation", "test"]
    })

    # Stats finales
    print("\n\n── FINAL STATS ──────────────────────────────────────")
    for split, ds in processed.items():
        est_h = len(ds) * 4.5 / 3600
        print(f"  {split:12s}: {len(ds):>6,} samples  (~{est_h:.2f}h)")

    # Vérification texte
    print("\n── SAMPLE SENTENCES AFTER NORMALISATION (train) ────")
    sentences = processed["train"]["sentence"]
    for i in range(5):
        print(f"  [{i}] {sentences[i]}")

    # Sauvegarde
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"\nSaving to {PROCESSED_DIR}...")
    processed.save_to_disk(PROCESSED_DIR)
    print("✓ Done. Next step: run 03_train.py")


if __name__ == "__main__":
    main()
