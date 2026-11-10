"""Shared helpers: seeding, metrics, and the append-only run log.

Every training run appends exactly one row to ``experiments/runs.csv``.
That file — not the README — is the source of truth for results.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Columns of experiments/runs.csv. Do not reorder without updating
# experiments/README.md.
RUN_COLUMNS = [
    "run_id",
    "date",
    "dataset",
    "model",
    "mode",              # "full" or "lora"
    "lora_r",            # empty for full fine-tuning
    "lora_alpha",
    "epochs",
    "train_examples",
    "batch_size",
    "learning_rate",
    "seed",
    "trainable_params",
    "total_params",
    "accuracy",
    "f1_macro",
    "train_time_sec",
    "device",
    "notes",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA, if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Metric function for the HF Trainer: accuracy + macro-F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def count_params(model: Any) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_run(csv_path: str | os.PathLike, row: dict[str, Any]) -> None:
    """Append one run row, creating the file + header if needed."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RUN_COLUMNS})
