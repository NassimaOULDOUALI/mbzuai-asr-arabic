"""
04_train.py
===========
LoRA fine-tuning of Whisper-small on Common Voice 18 Arabic.
Institute of Foundation Models — MBZUAI

Design decisions (motivated by literature)
-------------------------------------------
* Seq2SeqTrainer + PEFT LoRA: standard approach in the community
  (Gandhi et al. 2023, LoRA-Whisper Interspeech 2024, Pashto ASR 2025).

* Validation: fixed 1,000-sample subset of the val set (seed=42) is used
  during training for early stopping decisions. The full 10,471-sample
  test set is reserved exclusively for 05_evaluate_finetuned.py.
  Rationale: evaluating on all 10,471 samples every epoch would cost
  ~20 min per eval pass, making 5 epochs infeasible within the time budget.

* Primary metric: normalised WER (after normalize_arabic()).
  The Pashto ASR paper (arXiv:2604.06507) shows that training loss and WER
  can diverge — the checkpoint minimising loss may not minimise WER.
  We therefore use val WER as the early-stopping criterion and track
  both metrics separately in WandB.

* All checkpoints saved (one per epoch) as requested, plus the best
  checkpoint is flagged via load_best_model_at_end=True.

* Memory budget: verified to stay within 16 GB VRAM (technical test
  constraint) despite running on A100:
    Whisper-small fp16   : ~490 MB
    LoRA weights         : ~30 MB
    Activations (bs=16)  : ~5-6 GB
    AdamW states (LoRA)  : ~60 MB
    Total                : ~6-7 GB

* WandB: logs train_loss, eval_loss, eval_wer, eval_cer, learning_rate,
  gpu_memory_mb at every evaluation step. Run name includes timestamp
  for reproducibility.

Usage
-----
    # Optional: set your WandB project
    export WANDB_PROJECT="mbzuai-asr-arabic"
    python 04_train.py

    # To disable WandB and run offline:
    export WANDB_MODE=offline
    python 04_train.py

Requirements
------------
    pip install transformers datasets peft jiwer wandb torch accelerate
    pip install soundfile librosa
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import random
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ.setdefault("WANDB_PROJECT", "mbzuai-asr-arabic")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import wandb

from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
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
# CONFIG — all hyperparameters in one place
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "processed_dir"  : "./data/processed",
    "output_dir"     : "./checkpoints",
    "results_dir"    : "./results",

    # ── Model ─────────────────────────────────────────────────────────────────
    "model_name"     : "openai/whisper-small",
    "language"       : "arabic",
    "task"           : "transcribe",
    "sampling_rate"  : 16_000,

    # ── LoRA (Hu et al., ICLR 2022) ───────────────────────────────────────────
    # r=16, alpha=32 as reported in the technical report.
    # Target q_proj + v_proj: standard choice for Whisper LoRA
    # (LoRA-Whisper, Interspeech 2024; Pashto ASR, arXiv:2604.06507).
    "lora_r"         : 16,
    "lora_alpha"     : 32,
    "lora_dropout"   : 0.05,
    "lora_target"    : ["q_proj", "v_proj"],

    # ── Training ──────────────────────────────────────────────────────────────
    "num_epochs"            : 5,
    "per_device_train_batch": 16,
    "gradient_accumulation" : 2,       # effective batch = 32
    "learning_rate"         : 1e-4,
    "warmup_steps"          : 100,
    "lr_scheduler"          : "linear",
    "weight_decay"          : 0.0,
    "max_grad_norm"         : 1.0,
    "fp16"                  : True,
    "gradient_checkpointing": False,  # disabled: incompatible with LoRA grad graph
    "seed"                  : 42,

    # ── Evaluation ────────────────────────────────────────────────────────────
    # Evaluate once per epoch; val subset used for speed.
    # Full test set reserved for 05_evaluate_finetuned.py.
    "val_subset_size"       : 1000,    # samples from val set for training evals
    "per_device_eval_batch" : 16,
    "generation_max_length" : 225,     # Whisper max output tokens
    "early_stopping_patience": 3,      # epochs without val WER improvement

    # ── WandB ─────────────────────────────────────────────────────────────────
    "wandb_project"  : "mbzuai-asr-arabic",
    "wandb_run_name" : f"whisper-small-lora-arabic-{datetime.now().strftime('%Y%m%d_%H%M')}",
}


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation  (identical to 02_preprocess.py and 03_evaluate_baselines.py)
# ══════════════════════════════════════════════════════════════════════════════

KEEP_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]"
)

def normalize_arabic(text: str) -> str:
    """
    Normalisation pipeline shared across 02_preprocess.py,
    03_evaluate_baselines.py, and this script.
    Must remain identical across all files.
    """
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)   # strip tashkeel
    text = re.sub(r"[إأٱآا]", "ا", text)                # alef variants
    text = re.sub(r"[ىئ]",    "ي", text)                # ya variants
    text = re.sub(r"ة",        "ه", text)                # ta marbuta
    text = re.sub(r"ؤ",        "و", text)                # waw hamza
    text = KEEP_ARABIC.sub("", text)                      # strip non-arabic
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Data collator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WhisperDataCollator:
    """
    Collate audio features and tokenised labels for Seq2SeqTrainer.

    Input features are padded to 30 s (Whisper's fixed context window).
    Label padding uses -100 so CrossEntropyLoss ignores those positions.
    """
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:

        # ── Audio features ────────────────────────────────────────────────────
        input_feats = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_feats, return_tensors="pt"
        )

        # ── Labels ────────────────────────────────────────────────────────────
        # NOTE: we use "input_ids" as key only for the tokenizer.pad() call
        # (which requires that exact key), then immediately extract the tensor
        # and do NOT put input_ids into the final batch.
        # This avoids the "multiple values for keyword argument 'input_ids'"
        # error that occurs with PEFT >= 0.15 + transformers >= 5.x, where
        # PEFT passes input_ids to the decoder internally and collides with
        # any input_ids present in the batch dict.
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_feats, return_tensors="pt"
        )
        # Replace padding token id with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove BOS token if present (Whisper decoder handles it internally)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        # Only "input_features" and "labels" go into the batch —
        # no "input_ids" key to avoid the PEFT collision
        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(batch, processor: WhisperProcessor):
    """
    Convert raw audio arrays to log-Mel spectrograms and tokenise sentences.
    Applied via dataset.map() before training.
    """
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Compute metrics (called by Seq2SeqTrainer at each eval step)
# ══════════════════════════════════════════════════════════════════════════════

def build_compute_metrics(processor: WhisperProcessor):
    """
    Returns a compute_metrics function that:
      1. Decodes predicted token ids → strings
      2. Applies normalize_arabic() to both predictions and references
      3. Computes WER and CER with jiwer
      4. Logs everything to WandB

    Design note: normalisation is applied here (not just at final eval) so
    that early stopping decisions are made on the same normalised WER that
    will appear in the final evaluation table — consistent with the approach
    of Talafha et al. (Interspeech 2023) and the Pashto ASR paper.
    """
    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 padding back to pad_token_id before decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode to strings
        pred_strs  = processor.tokenizer.batch_decode(
            pred_ids,  skip_special_tokens=True
        )
        label_strs = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Normalise both sides — identical pipeline to preprocessing
        pred_norm  = [normalize_arabic(p) for p in pred_strs]
        label_norm = [normalize_arabic(l) for l in label_strs]

        # Filter pairs where reference is empty after normalisation
        pairs = [
            (p, r) for p, r in zip(pred_norm, label_norm)
            if len(r.strip()) > 0
        ]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}

        preds = [p for p, _ in pairs]
        refs  = [r for _, r in pairs]

        wer_score = compute_wer(refs, preds)
        cer_score = compute_cer(refs, preds)

        # Log GPU memory for monitoring
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / 1e6

        metrics = {
            "wer"       : round(wer_score * 100, 4),
            "cer"       : round(cer_score * 100, 4),
            "gpu_mem_mb": round(gpu_mb, 1),
        }
        return metrics

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# LoRA setup
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora(model, cfg: dict):
    """
    Inject LoRA adapters into Whisper-small.

    Target modules: q_proj and v_proj in all attention layers of both
    encoder and decoder. This is the configuration reported in the
    technical report and consistent with LoRA-Whisper (Song et al.,
    Interspeech 2024).

    Trainable parameters: ~3.9M / 244M (1.6%).
    """
    lora_config = LoraConfig(
        r             = cfg["lora_r"],
        lora_alpha    = cfg["lora_alpha"],
        lora_dropout  = cfg["lora_dropout"],
        target_modules= cfg["lora_target"],
        bias          = "none",
        task_type     = TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Custom Trainer — fixes PEFT + Whisper Seq2Seq incompatibility
# ══════════════════════════════════════════════════════════════════════════════

class WhisperLoRATrainer(Seq2SeqTrainer):
    """
    Subclass of Seq2SeqTrainer that:
    1. Removes unsupported kwargs from Whisper forward (input_ids)
    2. Injects forced_decoder_ids during generation so Whisper
       generates in Arabic — without this, WER is ~100% because
       the model generates in a random language.
    """

    def __init__(self, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.whisper_processor = processor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("input_ids", None)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        """
        Full override of prediction_step to fix two issues:
        1. 'labels' must not be passed to model.generate()
        2. forced_decoder_ids must be set so Whisper generates Arabic

        We implement generate + loss computation manually instead of
        calling super() which passes labels to generate() in transformers
        4.45.x, causing: ValueError: model_kwargs not used: ['labels']
        """
        inputs.pop("input_ids", None)

        # Extract labels — needed for loss + metrics but NOT for generate
        labels = inputs.get("labels", None)

        # Set forced_decoder_ids for Arabic generation
        if self.whisper_processor is not None:
            forced = self.whisper_processor.get_decoder_prompt_ids(
                language="arabic", task="transcribe"
            )
        else:
            forced = None

        has_labels = labels is not None
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # ── Generation (no labels) ────────────────────────────────────
            gen_inputs = {
                k: v for k, v in inputs.items()
                if k not in ("labels",)
            }
            generated_tokens = model.generate(
                **gen_inputs,
                forced_decoder_ids=forced,
                max_new_tokens=self.args.generation_max_length or 225,
            )

            # ── Loss (with labels, no generate) ───────────────────────────
            loss = None
            if has_labels:
                with self.compute_loss_context_manager():
                    loss_out = model(**inputs)
                    if isinstance(loss_out, dict):
                        loss = loss_out["loss"].mean().detach()
                    else:
                        loss = loss_out[0].mean().detach()

        # Pad generated_tokens to same length as labels for compute_metrics
        if labels is not None:
            max_len = max(generated_tokens.shape[-1], labels.shape[-1])
            pad_id = self.whisper_processor.tokenizer.pad_token_id if self.whisper_processor else 50256
            if generated_tokens.shape[-1] < max_len:
                import torch.nn.functional as F
                generated_tokens = F.pad(
                    generated_tokens,
                    (0, max_len - generated_tokens.shape[-1]),
                    value=pad_id
                )

        return (loss, generated_tokens, labels)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("  Arabic ASR — LoRA Fine-Tuning (Whisper-small)")
    log.info("=" * 65)
    log.info("Config: %s", json.dumps(CFG, indent=2, default=str))

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        log.warning("No GPU — training will be extremely slow.")

    # ── WandB initialisation ──────────────────────────────────────────────────
    wandb.init(
        project = CFG["wandb_project"],
        name    = CFG["wandb_run_name"],
        config  = CFG,
        tags    = ["whisper-small", "lora", "arabic", "cv18"],
    )
    log.info("WandB run: %s", wandb.run.url)

    # ── Processor & model ─────────────────────────────────────────────────────
    log.info("Loading processor and model: %s", CFG["model_name"])
    processor = WhisperProcessor.from_pretrained(
        CFG["model_name"],
        language = CFG["language"],
        task     = CFG["task"],
    )
    # Load in fp32 — the Trainer handles fp16 casting via fp16=True in
    # Seq2SeqTrainingArguments. Loading directly in fp16 disables gradients
    # on frozen weights, which breaks LoRA backward pass.
    model = WhisperForConditionalGeneration.from_pretrained(
        CFG["model_name"],
    )

    # Disable Whisper's default token suppression so LoRA can fully adapt
    model.config.forced_decoder_ids    = None
    model.config.suppress_tokens       = []
    model.generation_config.forced_decoder_ids = None

    # Apply LoRA
    model = apply_lora(model, CFG)

    # Force LoRA parameters to require gradients and set train mode.
    # Without this, gradient checkpointing or fp16 mixed precision can
    # silently detach the LoRA weight tensors from the compute graph.
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    log.info(
        "LoRA trainable params with grad: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    # Patch WhisperForConditionalGeneration.forward at class level.
    # PEFT injects kwargs that Whisper does not accept (input_ids,
    # inputs_embeds, etc.). We filter to only the kwargs Whisper
    # actually supports, inspected from its own signature.
    import inspect
    _whisper_fwd_params = set(
        inspect.signature(WhisperForConditionalGeneration.forward).parameters.keys()
    )
    _orig_whisper_fwd = WhisperForConditionalGeneration.forward
    def _patched_whisper_fwd(self_inner, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _whisper_fwd_params}
        return _orig_whisper_fwd(self_inner, *args, **kwargs)
    WhisperForConditionalGeneration.forward = _patched_whisper_fwd

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading preprocessed dataset from %s", CFG["processed_dir"])
    ds = load_from_disk(CFG["processed_dir"])

    train_ds = ds["train"]
    val_full  = ds["validation"]

    # Fixed val subset for training-time evaluation (speed)
    random.seed(CFG["seed"])
    val_indices = sorted(
        random.sample(range(len(val_full)), CFG["val_subset_size"])
    )
    val_ds = val_full.select(val_indices)
    log.info(
        "Train: %d samples  |  Val subset: %d / %d samples",
        len(train_ds), len(val_ds), len(val_full),
    )

    # Feature extraction — map to log-Mel + tokenise labels
    log.info("Extracting features (log-Mel spectrograms + tokenisation)...")
    train_ds = train_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=train_ds.column_names,
        num_proc=1,
        desc="train",
    )
    val_ds = val_ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=val_ds.column_names,
        num_proc=1,
        desc="val",
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = WhisperDataCollator(processor=processor)

    # ── Steps & eval frequency ────────────────────────────────────────────────
    steps_per_epoch = len(train_ds) // (
        CFG["per_device_train_batch"] * CFG["gradient_accumulation"]
    )
    total_steps = steps_per_epoch * CFG["num_epochs"]
    log.info(
        "Steps per epoch: %d  |  Total steps: %d", steps_per_epoch, total_steps
    )

    # Evaluate once per epoch
    eval_steps   = steps_per_epoch
    save_steps   = steps_per_epoch
    logging_steps = max(1, steps_per_epoch // 10)  # ~10 log points per epoch

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = CFG["output_dir"],
        num_train_epochs            = CFG["num_epochs"],
        per_device_train_batch_size = CFG["per_device_train_batch"],
        per_device_eval_batch_size  = CFG["per_device_eval_batch"],
        gradient_accumulation_steps = CFG["gradient_accumulation"],
        learning_rate               = CFG["learning_rate"],
        warmup_steps                = CFG["warmup_steps"],
        lr_scheduler_type           = CFG["lr_scheduler"],
        weight_decay                = CFG["weight_decay"],
        max_grad_norm               = CFG["max_grad_norm"],
        fp16                        = CFG["fp16"],
        gradient_checkpointing      = CFG["gradient_checkpointing"],
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = CFG["num_epochs"],  # keep all checkpoints
        logging_steps               = logging_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,              # lower WER is better
        predict_with_generate       = True,
        generation_max_length       = CFG["generation_max_length"],
        report_to                   = "wandb",
        run_name                    = CFG["wandb_run_name"],
        seed                        = CFG["seed"],
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WhisperLoRATrainer(
        processor       = processor,          # for forced_decoder_ids in eval
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = build_compute_metrics(processor),
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=CFG["early_stopping_patience"]
            )
        ],
        tokenizer        = processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    train_result = trainer.train()

    # ── Save best model & processor ───────────────────────────────────────────
    best_dir = Path(CFG["output_dir"]) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    log.info("Best model saved → %s", best_dir)

    # ── Log final training stats ───────────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── Save config for reproducibility ───────────────────────────────────────
    Path(CFG["results_dir"]).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(CFG["results_dir"]) / "train_config.json"
    with open(cfg_path, "w") as f:
        json.dump(CFG, f, indent=2, default=str)
    log.info("Config saved → %s", cfg_path)

    # ── WandB summary ─────────────────────────────────────────────────────────
    wandb.summary["best_val_wer"] = trainer.state.best_metric
    wandb.summary["total_steps"]  = trainer.state.global_step
    wandb.finish()

    log.info("")
    log.info("Training complete.")
    log.info("Best val WER : %.4f %%", trainer.state.best_metric)
    log.info("Best checkpoint : %s", trainer.state.best_model_checkpoint)
    log.info("Next step: run 05_evaluate_finetuned.py")


if __name__ == "__main__":
    main()
