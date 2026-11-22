# Reproducing System One (Jev-style) calibrated decisions for complaint categorization

A research project + course assignment: build a small **open** model that reproduces
the **System One** contract — take some state, answer typed questions, return a
decision *with a calibrated confidence* in a single pass — and test how close an open
model gets to the real thing on a real task.

## Research question

> Can a small **open** model reproduce System One (Jev-style) **calibrated typed-decision**
> behavior for consumer-complaint categorization, and how does it compare to
> (a) a prompted **frontier LLM** forced into the same typed-decision format and
> (b) a fine-tuned **encoder** baseline — on **accuracy, calibration (ECE), latency, and cost**?

The headline research texture is **calibration**: System One models claim well-calibrated
confidences. Whether a small fine-tuned open model can match that is a clean, measurable
question, so we report **reliability diagrams / ECE / Brier**, not just accuracy.

## Task & data

**Task.** Consumer complaint narrative → **Product** category (a typed decision over ~10
options). This is exactly the "discrete choice over a fixed option set" shape System One
models target.

**Data — important finding.** The CFPB has **removed consumer narratives from all of its
current public feeds** (the search API, the CSV export, and the bulk `complaints.csv.zip`
all now lack the narrative column — verified Nov 2026). Without narrative text there is no
classification task. We therefore use a **static Hugging Face snapshot captured while
narratives were still public**:

- **`milesbutler/consumer_complaints`** (MIT) — the full CFPB file as parquet, keeping both
  the `Product` label **and** the `Consumer Complaint` narrative text.

`src/data/prepare_data.py` downloads it, keeps rows with real narratives, consolidates the
older granular `Product` taxonomy into ~10 clean top-level classes (mapping documented in
`src/config.py`), caps per-class counts to tame the credit-reporting dominance, and writes
stratified `train/val/test` parquet.

## Approaches compared

| # | Approach | What it is | Role |
|---|----------|-----------|------|
| 1 | **System One repro** (ours) | Small Qwen, **frozen backbone + LoRA + a decision head** → probability over the option set in one pass; calibrated via temperature scaling / proper scoring | The model we build (satisfies the assignment) |
| 2 | **Encoder baseline** | DeBERTa classifier, single pass + temperature scaling | Strong, cheap, single-pass baseline |
| 3 | **Frontier LLM baseline** | Prompted hosted model via OpenRouter, forced into the same typed JSON `{label_id, confidence}` | "Is fine-tuning worth it?" |
| 4 | **Jev reference** | The **real** System One model (TypeSafe Jev) via OpenRouter's **Decisions API** — returns the choice + a calibrated probability distribution | Gold reference our open model approximates (callable, cannot be fine-tuned) |

The architecture for #1 follows the open "decision model" recipe (frozen LM + LoRA +
decision head) — see **Related work** below.

## Metrics

Accuracy, macro-F1, **ECE**, **MCE**, **Brier**, plus **latency (ms/decision)** and
**cost ($/1k decisions)**. All in `src/metrics.py` (includes `temperature_scale`).

## Repo layout

```
src/
  config.py              # paths, data source, label mapping, model names — single source of truth
  metrics.py             # ECE / MCE / Brier / reliability bins / temperature scaling
  data/prepare_data.py   # download snapshot -> clean -> consolidate labels -> stratified splits
  baselines/
    frontier_llm.py      # OpenRouter chat-LLM typed-decision baseline (accuracy/ECE/latency/cost)
    jev_baseline.py      # REAL Jev via OpenRouter Decisions API — the gold reference
    encoder.py           # fully fine-tuned DeBERTa classifier baseline + calibration
  sysone/
    train.py             # frozen Qwen + LoRA + decision head (the System One repro) + calibration
  report.py              # collate all outputs/*.json into one results table (md + csv)
data/processed/          # train/val/test.parquet + label_names.json (gitignored)
outputs/                 # metrics + predictions (gitignored)
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # then put your OpenRouter key in .env (already done locally)
```

## Running

> **On a rented GPU box?** Follow **[RUNBOOK.md](RUNBOOK.md)** — the full playbook (what to
> rent, how to get the code there, install, smoke test, full run, teardown, troubleshooting).
> The commands below are the same ones it uses.

```bash
# 1) Build the dataset (needs internet + `datasets`)
python -m src.data.prepare_data --max-per-class 8000

# 2) Frontier baseline (needs a network that can reach OpenRouter)
python -m src.baselines.frontier_llm --n 300 --model openai/gpt-4o-mini

# 3) System One reproduction (run on the rented GPU)
python -m src.sysone.train --base Qwen/Qwen2.5-0.5B --epochs 3 --bs 16
#   bigger backbone with 4-bit:  --base Qwen/Qwen2.5-1.5B --bs 8 --load-4bit

# 4) Encoder baseline (run on the rented GPU)
python -m src.baselines.encoder --base microsoft/deberta-v3-small --epochs 3 --bs 16

# 5) Jev reference (needs OpenRouter access; no GPU). Probe the response shape first:
python -m src.baselines.jev_baseline --probe
python -m src.baselines.jev_baseline --n 300 --workers 8
```

## Compute & network notes

- **Dev laptop GPU is 4 GB (RTX A1000)** — fine for data prep and the API baseline, **too
  small to fine-tune**. Training runs on a **rented GPU** (A10/A100/L4-class). Install the
  heavy deps (`torch`, `transformers`, `datasets`, `accelerate`, `peft`) there.
- **This machine's network blocks OpenRouter / Hugging Face / OpenAI** (TLS connections
  fail while Google works — a corporate firewall). Run the dataset download and the
  frontier baseline from the **rented box** or another network. The OpenRouter key is
  well-formed but **could not be verified here** for this reason — verify it on the box.

## Related work (to cite)

- *Releasing the Inner Jev: A Simple Recipe for Open Decision Models* (arXiv 2610.17205) —
  frozen LM + rank-32 LoRA + decision head. The recipe our model #1 follows.
- Open System One reproductions: **Decider** (Qwen3.5), **Kev** (Qwen, Apache-2.0),
  **Tev1** (Together AI).
- Calibration / audit: *Can the Confidence and Decisions of System 1 Models Be Trusted?*
  (arXiv 2611.12521); *Calibration and Risk-Budgeted Decisions with System One Models*
  (arXiv 2610.19811).
- Jev / System One background: TypeSafe AI (early access 2026-09-15); training via
  Reinforcement Learning for Calibrated Decisions (RLCD).

## Status

- [x] Project scaffold, metrics, config, `.env` (gitignored)
- [x] Data source identified & verified; `prepare_data.py` written
- [x] Frontier-LLM baseline written
- [x] **System One repro** written (`src/sysone/train.py`): frozen Qwen + LoRA + decision head + calibration
- [x] Encoder baseline written (`src/baselines/encoder.py`)
- [x] **Jev reference** written (`src/baselines/jev_baseline.py`): real Jev via OpenRouter Decisions API
- [ ] Run everything on the rented GPU / unblocked network (data prep + all 4 approaches)
- [ ] Results, reliability diagrams, paper write-up
