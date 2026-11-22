"""Central configuration — single source of truth for paths, data, models, and the task.

Research question
-----------------
Can a small OPEN model reproduce System One (Jev-style) calibrated typed-decision
behavior for consumer-complaint categorization, and how does it compare to
  (a) a prompted frontier LLM forced into the same typed-decision format, and
  (b) a fine-tuned encoder baseline,
on accuracy, calibration (ECE), latency, and cost?

Data note (important)
---------------------
The CFPB *removed consumer narratives from all of its current public feeds* (API,
CSV export, and the bulk zip no longer contain the narrative column). We therefore
use a STATIC Hugging Face snapshot captured while narratives were still public:
`milesbutler/consumer_complaints` — the full CFPB file (parquet, MIT) that keeps both
the `Product` label and the `Consumer Complaint` narrative text.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = ROOT / "checkpoints"

for _d in (RAW_DIR, PROC_DIR, OUTPUT_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------------------
HF_DATASET = "milesbutler/consumer_complaints"   # static snapshot w/ Product + narrative
TEXT_COL = "Consumer Complaint"                   # the narrative field in that snapshot
PRODUCT_COL = "Product"

MIN_NARRATIVE_CHARS = 40          # drop near-empty narratives
MAX_PER_CLASS = 8000              # cap per class to tame the credit-reporting dominance
                                  # (set to None to keep the natural distribution)
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --------------------------------------------------------------------------------------
# Label space: the snapshot uses the older granular taxonomy. We consolidate the ~18
# raw Product strings into a clean top-level set. This mapping is a documented
# preprocessing choice (several raw values are historical renames of the same product).
# --------------------------------------------------------------------------------------
PRODUCT_CONSOLIDATION = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting & repair",
    "Credit reporting": "Credit reporting & repair",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit card or prepaid card": "Credit or prepaid card",
    "Credit card": "Credit or prepaid card",
    "Prepaid card": "Credit or prepaid card",
    "Checking or savings account": "Bank account or service",
    "Bank account or service": "Bank account or service",
    "Money transfer, virtual currency, or money service": "Money transfer / service",
    "Money transfers": "Money transfer / service",
    "Virtual currency": "Money transfer / service",
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Student loan": "Student loan",
    "Payday loan, title loan, or personal loan": "Payday / personal / consumer loan",
    "Payday loan": "Payday / personal / consumer loan",
    "Consumer Loan": "Payday / personal / consumer loan",
    "Other financial service": "Other financial service",
}

# Canonical, ordered label list (index == class id). Derived from the mapping values.
LABELS = [
    "Credit reporting & repair",
    "Debt collection",
    "Mortgage",
    "Credit or prepaid card",
    "Bank account or service",
    "Money transfer / service",
    "Vehicle loan or lease",
    "Student loan",
    "Payday / personal / consumer loan",
    "Other financial service",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# --------------------------------------------------------------------------------------
# Frontier baseline (OpenRouter, OpenAI-compatible)
# --------------------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "openai/gpt-4o-mini")

# --------------------------------------------------------------------------------------
# System One reproduction + encoder baseline (trained on a RENTED GPU)
# --------------------------------------------------------------------------------------
SYS1_BASE_MODEL = os.environ.get("SYS1_BASE_MODEL", "Qwen/Qwen2.5-0.5B")  # frozen + LoRA + head
ENCODER_BASE_MODEL = os.environ.get("ENCODER_BASE_MODEL", "microsoft/deberta-v3-small")

SEED = 42
