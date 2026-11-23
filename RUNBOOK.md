# GPU-box runbook

The playbook for running the whole pipeline on a rented GPU. The laptop can't do this
(4 GB GPU + the SLB network blocks Hugging Face / OpenRouter). A normal cloud box has
neither limit. Budget: **one afternoon, a few dollars.**

---

## 0. What to rent

| Option | GPU | Good for | Rough cost |
|--------|-----|----------|-----------|
| Minimum | **T4 16 GB** (or free Colab T4) | DeBERTa + Qwen2.5-0.5B only | ~$0.20–0.40/hr |
| Mid | **L4 24 GB** or **A10 24 GB** | up to 1.5B–3B LoRA | ~$0.40–0.80/hr |
| **Recommended (our choice)** | **A40 48 GB** (or A6000 48 GB) | the **full 0.5B→7B sweep** in bf16 LoRA, no 4-bit | ~$0.40–0.80/hr |
| Overkill | A100 80 GB | only if you go past 7B | ~$1.50–3/hr |

**You do not need multiple GPUs.** Providers: RunPod, Lambda, Vast.ai, Paperspace.
Pick a **PyTorch 2.x / CUDA 12.x** base image so CUDA is already set up. The A40's 48 GB
holds a frozen 7B backbone (~14 GB bf16) plus the LoRA adapters and activations with
headroom, so **7B needs no 4-bit quantization** — run every size in plain bf16 LoRA.

Sanity check once you're on the box:
```bash
nvidia-smi          # confirm the GPU + VRAM
python --version    # 3.10–3.12
```

---

## 1. Get the code onto the box

**Option A — GitHub (recommended).** On the laptop, commit and push to a **private** repo
(`.env` is gitignored, so your key never leaves the laptop):
```bash
git add -A && git commit -m "scaffold"
git remote add origin git@github.com:<you>/cfpb-system-one.git
git push -u origin master
```
On the box:
```bash
git clone https://github.com/<you>/cfpb-system-one.git && cd cfpb-system-one
```

**Option B — direct upload.** `scp -r` the project folder (skip `data/`, `checkpoints/`,
`outputs/`), or use the provider's web upload / `runpodctl send`.

---

## 2. Environment + dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build
pip install -r requirements-gpu.txt
```

**On a network / persistent volume?** Point the Hugging Face cache at it so the model
weights (the 7B is ~15 GB) download **once** and survive pod restarts and re-attaching to a
new machine:
```bash
export HF_HOME=/persistent/hf        # match your volume's actual mount path
```
Add that line to `~/.bashrc` (or re-export it each session) so every run reuses the cache
instead of re-downloading. Keep the repo and `outputs/` on the volume too, and your results
survive even if you forget to copy before terminating.

Create `.env` **on the box** with your key (use a **rotated** key — the old one was pasted
in chat):
```bash
echo "OPENROUTER_API_KEY=sk-or-v1-YOUR-ROTATED-KEY" > .env
```

---

## 3. Build the dataset (once, ~2–5 min)

```bash
python -m src.data.prepare_data --max-per-class 8000
```
Writes `data/processed/{train,val,test}.parquet` + `label_names.json` and prints the class
distribution. (Needs internet for the Hugging Face snapshot — fine on a cloud box.)

---

## 4. Smoke test first (~5 min, catches setup bugs cheaply)

Run tiny versions of everything before the real runs:
```bash
python -m src.baselines.jev_baseline --probe                         # confirm Jev JSON shape
python -m src.baselines.jev_baseline --n 25 --workers 4
python -m src.baselines.frontier_llm --n 25 --model openai/gpt-4o-mini
python -m src.sysone.train   --base Qwen/Qwen2.5-0.5B --epochs 1 --bs 8 --train-size 2000
python -m src.baselines.encoder --base microsoft/deberta-v3-small --epochs 1 --train-size 2000
```
If all five finish and write to `outputs/`, you're green for the full run.

> **Jev probe:** `--probe` prints the raw response. The parser expects
> `answers.product.choice` and `answers.product.probabilities`. If the live field names
> differ, it's a one-line fix in `parse_answer` (`src/baselines/jev_baseline.py`).

---

## 5. Full run (produces the results table)

```bash
# 1) System One reproduction (ours) — ~1–2 h on the A40 for 0.5B
python -m src.sysone.train --base Qwen/Qwen2.5-0.5B --epochs 3 --bs 16
#    bigger backbone on the A40 (bf16 LoRA, no 4-bit needed):
#      --base Qwen/Qwen2.5-7B --bs 8 --grad-checkpoint

# 2) Encoder baseline — ~30–60 min
python -m src.baselines.encoder --base microsoft/deberta-v3-small --epochs 3 --bs 16

# 3) Jev reference — minutes, ~cents
python -m src.baselines.jev_baseline --n 1000 --workers 8

# 4) Frontier LLM baseline — minutes, a few cents
python -m src.baselines.frontier_llm --n 1000 --model openai/gpt-4o-mini
```

Each writes a JSON to `outputs/` with accuracy, macro-F1, **ECE/MCE/Brier**, latency, cost,
and reliability bins — one row per approach for your comparison table.

---

## 5a. Watch a run live (so a bad 7B run dies early, not after 8 h)

Training logs + evaluates every `--eval-steps` (default 200) and **early-stops** when val
macro-F1 stops improving (`--patience`). You get real signal within the first ~15–30 min.
The best checkpoint is saved every `--eval-steps`, so a crash costs at most that many steps.

Pick one way to watch:

**A. GPU utilization** (second terminal) — confirm the GPU is actually busy:
```bash
watch -n 5 nvidia-smi        # util high; VRAM steady (not creeping toward OOM)
```
**B. TensorBoard** (default, no account; logs under `checkpoints/sysone_<slug>/`):
```bash
tensorboard --logdir checkpoints --port 6006
# from the laptop:  ssh -L 6006:localhost:6006 <user>@<box>
#   (RunPod: expose TCP port 6006 and open its proxy URL)
```
Watch `eval/macro_f1` rise and `eval/ece` fall.

**C. Weights & Biases** (dashboard from anywhere, incl. phone):
```bash
pip install wandb && wandb login
python -m src.sysone.train --base Qwen/Qwen2.5-7B ... --report wandb
```

**When to kill it:** after 2–3 evals, `eval_macro_f1` should be clearly above chance
(~0.10 for 10 classes) and `train/loss` falling. Flat-at-chance or `loss=nan` after ~3
evals → Ctrl-C and fix (usually LR or data); early stopping also auto-stops a plateau.

---

## 5b. Compare model sizes (is a bigger model worth it?)

Train several backbones, then let the aggregator chart the scaling. Go small → large so you
get a cheap result first and can stop if the curve flattens:
```bash
for M in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B; do
  BS=16; [ "$M" = "Qwen/Qwen2.5-7B" ] && BS=8
  python -m src.sysone.train --base "$M" --epochs 3 --bs $BS --grad-checkpoint
done
python -m src.report        # -> outputs/results_table.md across all sizes + baselines
```

> **Disk during the sweep:** the four backbones together are ~25 GB of weights. With
> `HF_HOME` on your persistent volume they're cached once and reused — nothing to manage.
> On a small ephemeral disk instead, clear each size's weights before the next so you don't
> fill up:
> ```bash
> rm -rf "$HF_HOME"/hub/models--Qwen--Qwen2.5-0.5B    # (or ~/.cache/huggingface/hub/... if HF_HOME is unset)
> ```
> The LoRA checkpoints themselves are tiny (adapter only, a few hundred MB each), so
> `checkpoints/` stays small regardless.

Each backbone writes its own `outputs/sysone_<size>.json`, so the table shows accuracy /
macro-F1 / ECE / latency per size next to Jev, DeBERTa, and the frontier LLM — exactly the
"does 7B beat 1.5B enough to justify it?" comparison.

---

## 6. Get results off the box, then tear down

```bash
python -m src.report                 # build outputs/results_table.{md,csv}
# copy results back (JSON is small; skip the big checkpoints unless you want them)
tar czf outputs.tgz outputs/ && echo "scp this back to the laptop"
```
Then **stop or terminate the instance** so billing stops. `outputs/*.json` + the paper is
all you need next; re-renting to re-run is cheap.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA out of memory | lower `--bs`, add `--max-length 384`, or `--load-4bit` (needs bitsandbytes) |
| `bitsandbytes` import/CUDA error | drop `--load-4bit` (0.5B doesn't need it), or reinstall matching your CUDA |
| DeBERTa tokenizer error | `pip install sentencepiece` (it's in requirements-gpu.txt) |
| OpenRouter 401 | bad/expired key in `.env` — paste the rotated key, no quotes/spaces |
| Jev: no successful decisions | run `--probe`, compare the JSON to `parse_answer`, adjust field names |
| HF download slow | fine; the snapshot is a few hundred MB, one-time |

## Hyperparameters worth sweeping (if you have time)

- `--max-per-class` in data prep (class balance vs. data volume)
- System One: `--lora-r` {16,32,64}, `--epochs` {2,3,4}, backbone {0.5B, 1.5B, 3B, 7B}
- A **data-efficiency curve** (`--train-size` 500/1k/5k/all) is a cheap extra figure that
  strengthens the paper.
