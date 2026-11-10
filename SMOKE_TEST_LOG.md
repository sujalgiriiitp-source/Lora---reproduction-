# Smoke-test log — LoRA reproduction

## Status: NOT RUN — blocked, not passed

The end-to-end smoke test was **not completed**. Nothing below claims a
training run happened. `src/train.py` and `src/evaluate.py` are
syntax-checked (`py_compile` OK) and the notebooks are valid nbformat
JSON, but the pipeline has **not** been proven to run end to end yet.

## What was proven

**Dependency install — SUCCEEDED (10 Nov 2026, ~6 min).**

Command (run in `~/workspace/projects/lora-reproduction/`):

```bash
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch, transformers, peft, datasets, sklearn; print('IMPORTS_OK ...')"
```

Result:

```text
IMPORTS_OK torch= 2.15.0+cpu transformers= 4.57.6 peft= 0.21.2
```

Note: the PyTorch CPU wheel download hit three read timeouts
(`download-r2.pytorch.org`, 15 s read timeout) and retried before
succeeding. If a future install fails the same way, re-run the torch
install — the retries recovered on their own here.

## What was attempted for the smoke test — BLOCKED

Intended smoke test (tiny subset only — 200 train / 100 eval, 1 epoch,
DistilBERT, LoRA r=8, CPU, batch 8):

```bash
cd ~/workspace/projects/lora-reproduction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
.venv/bin/python src/train.py --mode lora --r 8 --epochs 1 \
  --max-train 200 --max-eval 100 --batch-size 8 \
  --output runs/smoke-lora-r8 --run-id smoke-lora-r8
```

Outcome, honestly recorded:

1. First attempt (background exec) sat in `pending_user_confirmation`
   for ~4 minutes with zero output — the command never started. It was
   stopped (SIGTERM) without running anything.
2. A retry exec was **declined** (tool error: "pending approval
   correlation failed closed … 7 conflicting pending approval
   identities", `userConfirmationStatus: declined`).
3. Per the denial, the smoke test was **not retried and not worked
   around**. No training ran, no checkpoint was produced, and no row
   was added to `experiments/runs.csv` (it still contains only the
   header).

## What remains

1. **Re-run the smoke test above** once the user explicitly asks to try
   again. Expected: a few minutes on CPU (model + SST-2 download on
   first run), ending with a `[result]` line and one `smoke-lora-r8`
   row in `experiments/runs.csv` labelled `smoke test`.
2. **Then run `src/evaluate.py`** on the smoke checkpoint to prove the
   eval path too:

   ```bash
   .venv/bin/python src/evaluate.py --checkpoint runs/smoke-lora-r8 \
     --mode lora --r 8 --max-eval 100 --batch-size 8
   ```

3. **Full runs belong on Colab (T4 GPU)** — see the README. Do not run
   the full 3-epoch SST-2 training on this VM's CPU.
