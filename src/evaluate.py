"""Evaluate a saved checkpoint (full or LoRA) on the validation split.

The checkpoint directory must contain the model + tokenizer saved by
``src/train.py``. For a LoRA checkpoint it also contains the adapter
weights; ``--mode lora --r <r>`` rebuilds the adapter wrapper before
evaluating. Prints accuracy / macro-F1 and does not retrain anything.

Example::

    python src/evaluate.py --checkpoint runs/lora-r8 --mode lora --r 8
    python src/evaluate.py --checkpoint runs/full --mode full
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from train import DATASET_CONFIGS, MODEL_NAME
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="sst2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_path, hf_config, text_col, label_col, num_labels = DATASET_CONFIGS[args.dataset]
    raw = load_dataset(hf_path, hf_config) if hf_config else load_dataset(hf_path)
    eval_ds = raw["validation"] if "validation" in raw else raw["test"]
    if args.max_eval:
        eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    if args.mode == "lora":
        # Adapter weights + config are stored in the checkpoint dir by
        # train.py; from_pretrained reads the saved config (r, alpha, ...).
        model = PeftModel.from_pretrained(model, args.checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint, num_labels=num_labels
        )
    model.to(device)
    model.eval()

    preds, labels = [], []
    for i in range(0, len(eval_ds), args.batch_size):
        batch = eval_ds[i:i + args.batch_size]
        inputs = tokenizer(
            batch[text_col], truncation=True, max_length=args.max_len,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(np.argmax(logits.cpu().numpy(), axis=-1).tolist())
        labels.extend(batch[label_col])

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"[eval] checkpoint={args.checkpoint} n={len(labels)} "
          f"accuracy={acc:.4f} f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
