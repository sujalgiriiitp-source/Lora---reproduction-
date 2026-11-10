# Experiments log

`runs.csv` is the append-only record of every training run in this study.
It is the source of truth: any number in the README or the report must
match a row here, or it does not get claimed.

## Columns

| Column | Meaning |
| --- | --- |
| `run_id` | Short label, e.g. `full`, `lora-r4`, `lora-r8`, `lora-r16` |
| `date` | ISO date of the run (YYYY-MM-DD) |
| `dataset` | `sst2` or `ag_news` — the dataset actually used |
| `model` | Base model, e.g. `distilbert-base-uncased` |
| `mode` | `full` (all weights) or `lora` (adapters only) |
| `lora_r` | LoRA rank; empty for full fine-tuning |
| `lora_alpha` | LoRA alpha; convention in this repo is `alpha = 2r` |
| `epochs` | Training epochs |
| `train_examples` | Number of training examples actually used |
| `batch_size` | Per-device training batch size |
| `learning_rate` | Learning rate used |
| `seed` | Random seed (42 unless stated otherwise) |
| `trainable_params` | Parameters updated during training |
| `total_params` | All parameters of the (adapted) model |
| `accuracy` | Validation accuracy (0–1) |
| `f1_macro` | Validation macro-F1 (0–1) |
| `train_time_sec` | Wall-clock training time in seconds |
| `device` | `cuda` or `cpu` |
| `notes` | e.g. `smoke test` — any deviation from the standard setup |

## Rules

1. One row per run, appended by `src/train.py` — never edit a row by hand.
   If a run was flawed, add a corrected run as a new row and note the flaw
   in `notes`; do not rewrite history.
2. A smoke-test row (few hundred examples, 1 epoch, CPU) is labelled as
   such in `notes`. It proves the pipeline runs; it is **not** a result
   and must never be copied into the results table.
3. The four planned full runs — `full`, `lora-r4`, `lora-r8`, `lora-r16` —
   must share seed, dataset, epochs, and max length. Only `mode` / `r`
   (and the learning rate, 2e-5 vs 1e-4, as documented in the README)
   may differ.
