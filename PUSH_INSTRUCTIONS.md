# Push instructions — `Lora---reproduction-`

Repo name: **`Lora---reproduction-`**
Owner: **`sujalgiriiitp-source`**
Local folder: `~/workspace/projects/lora-reproduction/`
Default branch: `main`

> I (Muse) did **not** create this repository on GitHub. A previous
> attempt to create a repo via the GitHub connector failed with
> `403 Resource not accessible by integration` (31 Oct 2026), so the repo
> must be created by Sujal first. Do not assume it exists until the
> GitHub page loads.

## Step 1 — Sujal creates the empty repo (2 minutes)

1. Open https://github.com/new
2. Owner: `sujalgiriiitp-source`
3. Repository name: `Lora---reproduction-`
4. Description: `Reproducing LoRA (Hu et al., 2021) on DistilBERT + SST-2, with a rank ablation study (r=4/8/16).`
5. Visibility: **Public** (professors need to read it without a login)
6. **Do NOT tick** "Add a README file", "Add .gitignore", or "Choose a
   license" — the local repo already contains all three, and an
   initialised remote would conflict with the first push.
7. Click **Create repository**. Leave the "quick setup" page open.

## Step 2 — Push from this machine

```bash
cd ~/workspace/projects/lora-reproduction
git remote add origin https://github.com/sujalgiriiitp-source/Lora---reproduction-.git
git branch -M main
git push -u origin main
```

If `git push` asks for credentials, stop — do not paste a token in chat
or in the terminal history of a shared screen. Use the GitHub website
upload or ask Muse to try the connector snapshot-push instead.

## Step 3 — Verify (do not skip)

1. Open https://github.com/sujalgiriiitp-source/Lora---reproduction-
2. Check: README renders, `src/`, `notebooks/`, `experiments/`,
   `report/` are all present, and the commit count is 1 (the first
   commit below).
3. Click a Colab badge in the README — it must open the notebook, not
   a 404. If it 404s, the repo name/branch in the badge URL is wrong.

## First commit (already made locally)

- Message: `Initial commit: LoRA reproduction (DistilBERT + SST-2, rank ablation r=4/8/16)`
- Contents: README, LICENSE (MIT), .gitignore, requirements.txt,
  `src/train.py`, `src/evaluate.py`, `src/utils.py`, 4 notebooks,
  `experiments/runs.csv` (header only) + README, `report/report.md`,
  this file.
- Deliberately **not** in this commit: `SMOKE_TEST_LOG.md` (written
  after the first commit; commit it once the smoke test has actually
  run).
- Deliberately **not** committed: `runs/` (checkpoints), `.venv/`,
  `__pycache__/`, downloaded datasets/models (see `.gitignore`).
