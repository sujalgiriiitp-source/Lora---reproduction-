"""Train DistilBERT on SST-2 (or AG News) with full fine-tuning or LoRA.

This is the single entry point for every run in the study, so that the
four runs differ only in ``--mode`` / ``--r`` and nothing else.

Examples
--------
Full fine-tuning, 3 epochs (the baseline)::

    python src/train.py --mode full --epochs 3 --output runs/full

LoRA, rank 8::

    python src/train.py --mode lora --r 8 --epochs 3 --output runs/lora-r8

Tiny CPU smoke test (proves the pipeline runs; NOT a result)::

    python src/train.py --mode lora --r 8 --epochs 1 \\
        --max-train 200 --max-eval 100 --batch-size 8 --output runs/smoke

Every run appends one row to ``experiments/runs.csv`` and saves the
trained model + tokenizer to ``--output``.
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics, count_params, log_run, set_seed

MODEL_NAME = "distilbert-base-uncased"
DATASET_CONFIGS = {
    # name -> (hf path, hf config, text column, label column, num labels)
    "sst2": ("nyu-mll/glue", "sst2", "sentence", "label", 2),
    "ag_news": ("ag_news", None, "text", "label", 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilBERT: full FT or LoRA.")
    parser.add_argument("--mode", choices=["full", "lora"], required=True,
                        help="'full' = update all weights; 'lora' = adapters only.")
    parser.add_argument("--r", type=int, default=8,
                        help="LoRA rank (only used when --mode lora). Default: 8.")
    parser.add_argument("--target-modules", default="q_lin,v_lin",
                        help="Comma-separated LoRA target modules. Default: "
                        "q_lin,v_lin (attention only). Use "
                        "q_lin,k_lin,v_lin,out_lin,lin1,lin2 for all linear layers.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="sst2",
                        help="sst2 (default) or ag_news (documented fallback).")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-device train batch size (use 8 on CPU).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate. Default: 2e-5 (full), 1e-4 (LoRA).")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap training examples (smoke tests only).")
    parser.add_argument("--max-eval", type=int, default=None,
                        help="Cap validation examples (smoke tests only).")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True,
                        help="Directory for the saved model + tokenizer.")
    parser.add_argument("--runs-csv", default="experiments/runs.csv",
                        help="Append-only run log.")
    parser.add_argument("--run-id", default=None,
                        help="Label for this run; defaults to <mode>[-r<R>].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    lr = args.lr if args.lr is not None else (2e-5 if args.mode == "full" else 1e-4)
    run_id = args.run_id or (args.mode if args.mode == "full" else f"lora-r{args.r}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] {run_id} | dataset={args.dataset} | mode={args.mode} "
          f"| r={args.r if args.mode == 'lora' else '-'} | lr={lr} | device={device}")

    # --- Data ---------------------------------------------------------
    hf_path, hf_config, text_col, label_col, num_labels = DATASET_CONFIGS[args.dataset]
    raw = load_dataset(hf_path, hf_config) if hf_config else load_dataset(hf_path)
    train_ds, eval_ds = raw["train"], raw["validation"] if "validation" in raw else raw["test"]
    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    if args.max_eval:
        eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=args.max_len)

    # NOTE (11 Nov 2026): remove_columns MUST stay [] — removing all columns
    # also drops the label, and the Trainer then fails with "The model did
    # not return a loss from the inputs, only logits". The Trainer drops
    # unused columns itself; the label column is kept here.
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=[])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=[])

    # --- Model --------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    if args.mode == "lora":
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        config = LoraConfig(
            r=args.r,
            lora_alpha=2 * args.r,      # paper convention: alpha / r scaling
            lora_dropout=0.1,
            target_modules=target_modules,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    trainable, total = count_params(model)
    print(f"[params] trainable={trainable:,} / total={total:,} "
          f"({100 * trainable / total:.2f}%)")

    # --- Training -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    metrics = trainer.evaluate()
    accuracy = float(metrics.get("eval_accuracy", 0.0))
    f1 = float(metrics.get("eval_f1_macro", 0.0))
    print(f"[result] accuracy={accuracy:.4f} f1_macro={f1:.4f} "
          f"train_time={train_time:.1f}s")

    # --- Save + log ---------------------------------------------------
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    notes_parts = []
    if args.max_train:
        notes_parts.append("smoke test")
    if args.mode == "lora" and args.target_modules != "q_lin,v_lin":
        notes_parts.append(f"target={args.target_modules}")
    log_run(args.runs_csv, {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "model": MODEL_NAME,
        "mode": args.mode,
        "lora_r": args.r if args.mode == "lora" else "",
        "lora_alpha": 2 * args.r if args.mode == "lora" else "",
        "epochs": args.epochs,
        "train_examples": len(train_ds),
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "seed": args.seed,
        "trainable_params": trainable,
        "total_params": total,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1, 4),
        "train_time_sec": round(train_time, 1),
        "device": device,
        "notes": "; ".join(notes_parts),
    })
    print(f"[log] row appended to {args.runs_csv}")


if __name__ == "__main__":
    main()
