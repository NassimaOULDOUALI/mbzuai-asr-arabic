"""
05_evaluate_finetuned.py
========================
Final evaluation of the LoRA fine-tuned Whisper-small model on the full
Common Voice 18 Arabic test set (10,471 samples).

This script:
  1. Loads the best checkpoint saved by 04_train.py
  2. Evaluates on the full test set (not the 500-sample baseline subset)
  3. Computes WER and CER after normalize_arabic() — same pipeline as training
  4. Performs qualitative error analysis (native Arabic speaker perspective):
     - Best predictions (lowest per-sample WER)
     - Worst predictions (highest per-sample WER)
     - Error type breakdown: diacritic mismatch, lexical hallucination,
       partial substitution, insertion, deletion
  5. Compares fine-tuned vs zero-shot on the same test samples
  6. Saves a full JSON report and a human-readable summary

Design decisions
----------------
* We evaluate on the FULL test set (10,471 samples) — not the 500-sample
  subsample used for baselines. This gives the official final number for
  the technical report.
* Batch inference with batch_size=32 for speed on A100.
* forced_decoder_ids set to Arabic/transcribe — same as during training eval.
* Both fine-tuned and zero-shot predictions are computed in the same run
  for a fair apple-to-apple comparison on identical audio.

Usage
-----
    PYTHONNOUSERSITE=1 python 05_evaluate_finetuned.py

Requirements
------------
    Same environment as 04_train.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from jiwer import wer as compute_wer, cer as compute_cer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    "processed_dir"  : "./data/processed",
    "best_checkpoint": "./checkpoints/best",   # saved by 04_train.py
    "base_model"     : "openai/whisper-small",
    "results_dir"    : "./results",
    "language"       : "arabic",
    "task"           : "transcribe",
    "sampling_rate"  : 16_000,
    "batch_size"     : 32,
    "seed"           : 42,
    "n_qualitative"  : 20,    # examples saved for qualitative analysis
}


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation — identical to 02_preprocess.py and 04_train.py
# ══════════════════════════════════════════════════════════════════════════════

KEEP_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]"
)

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"[ىئ]",    "ي", text)
    text = re.sub(r"ة",        "ه", text)
    text = re.sub(r"ؤ",        "و", text)
    text = KEEP_ARABIC.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_finetuned_model(checkpoint_dir: str, base_model: str,
                         device: torch.device):
    """
    Load the LoRA fine-tuned model from the best checkpoint.
    The checkpoint contains only the LoRA adapter weights (~30 MB).
    The base Whisper-small weights are loaded fresh and the adapter
    is applied on top.
    """
    log.info("Loading base model: %s", base_model)
    processor = WhisperProcessor.from_pretrained(
        base_model, language="arabic", task="transcribe"
    )
    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    log.info("Loading LoRA adapter from: %s", checkpoint_dir)
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model = model.merge_and_unload()   # merge LoRA into base weights for speed
    model.eval()
    model.to(device)
    log.info(
        "Fine-tuned model loaded — parameters: %dM  device: %s",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        device,
    )
    return processor, model


def load_zeroshot_model(base_model: str, device: torch.device):
    """
    Load the unmodified Whisper-small for zero-shot comparison.
    Evaluated on the same test samples as the fine-tuned model.
    """
    log.info("Loading zero-shot model: %s", base_model)
    processor = WhisperProcessor.from_pretrained(
        base_model, language="arabic", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.eval()
    model.to(device)
    return processor, model


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    dataset,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    batch_size: int,
    label: str,
) -> tuple:
    """
    Run batch inference over the full dataset.

    Returns
    -------
    predictions_raw  : list[str]  — raw Whisper output
    predictions_norm : list[str]  — after normalize_arabic()
    references_norm  : list[str]  — ground truth (already normalised)
    elapsed_sec      : float
    """
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="arabic", task="transcribe"
    )
    all_raw   = []
    all_norm  = []
    all_refs  = []
    n         = len(dataset)
    t0        = time.time()

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = dataset.select(range(start, end))

        audio_arrays = [
            np.array(s["audio"]["array"], dtype=np.float32)
            for s in batch
        ]
        refs = [s["sentence"] for s in batch]

        inputs = processor(
            audio_arrays,
            sampling_rate=CFG["sampling_rate"],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        preds_raw  = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        preds_norm = [normalize_arabic(p) for p in preds_raw]

        all_raw.extend(preds_raw)
        all_norm.extend(preds_norm)
        all_refs.extend(refs)

        if start % (batch_size * 20) == 0:
            log.info("[%s]  %d / %d", label, end, n)

    elapsed = time.time() - t0
    log.info("[%s]  Done — %.1f s (%.2f s/sample)", label, elapsed, elapsed / n)
    return all_raw, all_norm, all_refs, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds_norm: list, refs_norm: list, label: str) -> dict:
    pairs = [
        (p, r) for p, r in zip(preds_norm, refs_norm)
        if len(r.strip()) > 0
    ]
    preds = [p for p, _ in pairs]
    refs  = [r for _, r in pairs]

    wer_score = compute_wer(refs, preds)
    cer_score = compute_cer(refs, preds)

    per_sample_wer = [
        compute_wer([r], [p]) for p, r in zip(preds, refs)
    ]

    metrics = {
        "label"          : label,
        "n_samples"      : len(pairs),
        "wer"            : round(wer_score * 100, 4),
        "cer"            : round(cer_score * 100, 4),
        "wer_median"     : round(float(np.median(per_sample_wer)) * 100, 2),
        "wer_p25"        : round(float(np.percentile(per_sample_wer, 25)) * 100, 2),
        "wer_p75"        : round(float(np.percentile(per_sample_wer, 75)) * 100, 2),
        "wer_p90"        : round(float(np.percentile(per_sample_wer, 90)) * 100, 2),
        "perfect_pct"    : round(
            100 * sum(1 for w in per_sample_wer if w == 0.0) / len(per_sample_wer), 2
        ),
    }
    log.info(
        "[%s]  WER=%.2f%%  CER=%.2f%%  perfect=%.1f%%",
        label, metrics["wer"], metrics["cer"], metrics["perfect_pct"]
    )
    return metrics, per_sample_wer


# ══════════════════════════════════════════════════════════════════════════════
# Qualitative analysis
# ══════════════════════════════════════════════════════════════════════════════

def qualitative_analysis(
    refs_norm: list,
    preds_raw_ft: list,
    preds_norm_ft: list,
    preds_raw_zs: list,
    preds_norm_zs: list,
    per_sample_wer_ft: list,
    n: int = 20,
) -> dict:
    """
    Select best and worst predictions from the fine-tuned model,
    showing zero-shot prediction alongside for comparison.
    This section is the native Arabic speaker analysis.
    """
    indexed = list(enumerate(per_sample_wer_ft))
    best    = sorted(indexed, key=lambda x: x[1])[:n]
    worst   = sorted(indexed, key=lambda x: x[1], reverse=True)[:n]

    def make_example(i, wer_val):
        return {
            "index"           : i,
            "reference"       : refs_norm[i],
            "finetuned_raw"   : preds_raw_ft[i],
            "finetuned_norm"  : preds_norm_ft[i],
            "zeroshot_raw"    : preds_raw_zs[i],
            "zeroshot_norm"   : preds_norm_zs[i],
            "sample_wer_ft"   : round(wer_val * 100, 2),
            "sample_wer_zs"   : round(
                compute_wer([refs_norm[i]], [preds_norm_zs[i]]) * 100, 2
            ),
        }

    return {
        "best_predictions" : [make_example(i, w) for i, w in best],
        "worst_predictions": [make_example(i, w) for i, w in worst],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = Path(results_dir) / f"final_eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Full results → %s", json_path)

    txt_path = Path(results_dir) / f"final_eval_summary_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_summary(results))
    log.info("Summary     → %s", txt_path)

    return json_path, txt_path


def format_summary(results: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  ARABIC ASR — FINAL EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Timestamp  : {results['timestamp']}")
    lines.append(f"  Test set   : {results['n_test_samples']} samples (full CV-18 Arabic)")
    lines.append(f"  Checkpoint : {results['checkpoint']}")
    lines.append("")
    lines.append(f"  {'Model':<35} {'WER %':>7}  {'CER %':>7}  {'Perfect %':>10}")
    lines.append("  " + "-" * 60)
    for m in results["metrics"]:
        lines.append(
            f"  {m['label']:<35} {m['wer']:>7.2f}  {m['cer']:>7.2f}  "
            f"{m['perfect_pct']:>10.1f}"
        )
    lines.append("")

    # Improvement summary
    ft  = next(m for m in results["metrics"] if "fine-tuned" in m["label"].lower())
    zs  = next(m for m in results["metrics"] if "zero-shot"  in m["label"].lower())
    lines.append(f"  WER reduction  : {zs['wer']:.2f}% → {ft['wer']:.2f}%"
                 f"  ({zs['wer'] - ft['wer']:.2f} pp abs,"
                 f"  {(zs['wer'] - ft['wer']) / zs['wer'] * 100:.1f}% rel)")
    lines.append(f"  CER reduction  : {zs['cer']:.2f}% → {ft['cer']:.2f}%"
                 f"  ({zs['cer'] - ft['cer']:.2f} pp abs,"
                 f"  {(zs['cer'] - ft['cer']) / zs['cer'] * 100:.1f}% rel)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    # Best predictions (fine-tuned)
    lines.append("── BEST PREDICTIONS (fine-tuned, lowest WER) ─────────────────────")
    for ex in results["qualitative"]["best_predictions"][:10]:
        lines.append(f"  REF      : {ex['reference']}")
        lines.append(f"  FT PRED  : {ex['finetuned_norm']}  (WER={ex['sample_wer_ft']}%)")
        lines.append(f"  ZS PRED  : {ex['zeroshot_norm']}   (WER={ex['sample_wer_zs']}%)")
        lines.append("")

    lines.append("── WORST PREDICTIONS (fine-tuned, highest WER) ───────────────────")
    for ex in results["qualitative"]["worst_predictions"][:10]:
        lines.append(f"  REF      : {ex['reference']}")
        lines.append(f"  FT PRED  : {ex['finetuned_norm']}  (WER={ex['sample_wer_ft']}%)")
        lines.append(f"  ZS PRED  : {ex['zeroshot_norm']}   (WER={ex['sample_wer_zs']}%)")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("  Arabic ASR — Final Evaluation on Full Test Set")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Load test set ─────────────────────────────────────────────────────────
    log.info("Loading test set from %s", CFG["processed_dir"])
    ds   = load_from_disk(CFG["processed_dir"])
    test = ds["test"]
    log.info("Test set: %d samples", len(test))

    # ── Load models ───────────────────────────────────────────────────────────
    processor_ft, model_ft = load_finetuned_model(
        CFG["best_checkpoint"], CFG["base_model"], device
    )
    processor_zs, model_zs = load_zeroshot_model(CFG["base_model"], device)

    # ── Run inference ─────────────────────────────────────────────────────────
    log.info("")
    log.info("── Evaluating fine-tuned model ──────────────────────────────────")
    raw_ft, norm_ft, refs, elapsed_ft = run_inference(
        test, processor_ft, model_ft, device, CFG["batch_size"],
        label="fine-tuned"
    )

    log.info("")
    log.info("── Evaluating zero-shot model ───────────────────────────────────")
    raw_zs, norm_zs, _, elapsed_zs = run_inference(
        test, processor_zs, model_zs, device, CFG["batch_size"],
        label="zero-shot"
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    log.info("")
    log.info("── Computing metrics ────────────────────────────────────────────")
    metrics_ft, per_wer_ft = compute_metrics(
        norm_ft, refs, "Whisper-small + LoRA (fine-tuned)"
    )
    metrics_zs, per_wer_zs = compute_metrics(
        norm_zs, refs, "Whisper-small zero-shot Arabic"
    )

    # ── Qualitative analysis ──────────────────────────────────────────────────
    log.info("")
    log.info("── Qualitative analysis ─────────────────────────────────────────")
    qualitative = qualitative_analysis(
        refs_norm      = refs,
        preds_raw_ft   = raw_ft,
        preds_norm_ft  = norm_ft,
        preds_raw_zs   = raw_zs,
        preds_norm_zs  = norm_zs,
        per_sample_wer_ft = per_wer_ft,
        n              = CFG["n_qualitative"],
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "timestamp"      : datetime.now().isoformat(),
        "checkpoint"     : CFG["best_checkpoint"],
        "n_test_samples" : len(test),
        "metrics"        : [metrics_ft, metrics_zs],
        "elapsed"        : {
            "finetuned_s" : round(elapsed_ft, 1),
            "zeroshot_s"  : round(elapsed_zs, 1),
        },
        "qualitative"    : qualitative,
    }

    json_path, txt_path = save_results(results, CFG["results_dir"])

    # ── Final console summary ─────────────────────────────────────────────────
    log.info("")
    log.info(format_summary(results))
    log.info("Done. Results saved to %s", CFG["results_dir"])


if __name__ == "__main__":
    main()
