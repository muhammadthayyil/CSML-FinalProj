"""Collate every result in outputs/ into one comparison table (markdown + CSV).

Run after any mix of runs (it just reads whatever JSONs exist):
    python -m src.report

Handles all four result shapes:
  - sysone_*.json / encoder_*.json  -> uses the temperature-CALIBRATED metrics
  - frontier_*.json                 -> metrics nested under "summary"
  - jev_*.json                      -> metrics at top level
Writes outputs/results_table.md and outputs/results_table.csv. With several
sysone_<backbone>.json files, you see the accuracy/calibration-vs-size curve directly.
"""
from __future__ import annotations

import csv
import json

from src import config


def _load(path):
    obj = json.loads(path.read_text("utf-8"))
    if "summary" in obj:                       # frontier_*.json nests under "summary"
        obj = obj["summary"]
    if "metrics_calibrated" in obj:            # trained models: report calibrated metrics
        m = obj["metrics_calibrated"]
    elif "metrics" in obj:                     # API baselines (frontier / jev)
        m = obj["metrics"]
    else:
        return None
    return {
        "name": obj.get("model") or obj.get("base_model") or path.stem,
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "ece": m.get("ece"),
        "brier": m.get("brier"),
        "latency_ms": obj.get("latency_ms_mean"),
        "cost_per_1k": obj.get("cost_per_1k_usd"),
        "n": m.get("n") or obj.get("n_evaluated"),
    }


def main():
    rows = []
    for p in sorted(config.OUTPUT_DIR.glob("*.json")):
        if p.stem == "results_table":
            continue
        r = _load(p)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No result JSONs in {config.OUTPUT_DIR}. Run training/baselines first.")

    rows.sort(key=lambda r: (r["macro_f1"] is None, -(r["macro_f1"] or 0)))
    cols = ["name", "accuracy", "macro_f1", "ece", "brier", "latency_ms", "cost_per_1k", "n"]

    def fmt(v):
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    md_text = "# Results (sorted by macro-F1)\n\n" + "\n".join(md) + "\n"
    (config.OUTPUT_DIR / "results_table.md").write_text(md_text, "utf-8")

    with open(config.OUTPUT_DIR / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    print(md_text)
    print("Wrote", config.OUTPUT_DIR / "results_table.md", "and results_table.csv")


if __name__ == "__main__":
    main()
