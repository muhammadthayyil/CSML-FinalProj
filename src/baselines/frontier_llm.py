"""Frontier-LLM baseline: prompt a hosted model (via OpenRouter) into the SAME
typed-decision contract as our System One model — pick one product category and
report a calibrated confidence — then score accuracy, ECE, latency, and cost.

This is the "is fine-tuning even worth it?" comparison point.

Run (on a network that can reach OpenRouter, e.g. the rented GPU box):
    pip install openai python-dotenv pandas scikit-learn scipy
    python -m src.baselines.frontier_llm --n 200 --model openai/gpt-4o-mini

Reads OPENROUTER_API_KEY from .env. Writes outputs/frontier_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # populate OPENROUTER_API_KEY before config reads the environment

from src import config          # noqa: E402
from src.metrics import evaluate  # noqa: E402

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_messages(text: str, labels: list[str]):
    numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(labels))
    system = (
        "You are a precise classifier for U.S. consumer-finance complaints. "
        "Read the complaint and choose the single best product category. "
        "Then give a calibrated confidence in [0,1] that your choice is correct "
        "(use the full range; do not default to 1.0)."
    )
    user = (
        f"Product categories:\n{numbered}\n\n"
        f'Complaint:\n"""\n{text[:4000]}\n"""\n\n'
        f'Respond with ONLY JSON: {{"label_id": <int 0-{len(labels) - 1}>, '
        f'"confidence": <float 0-1>}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_reply(content: str, n_labels: int):
    """Return (label_id, confidence) or (None, None) on failure."""
    if not content:
        return None, None
    m = _JSON_RE.search(content)
    blob = m.group(0) if m else content
    try:
        obj = json.loads(blob)
        lid = int(obj["label_id"])
        conf = float(obj["confidence"])
    except Exception:
        return None, None
    if not (0 <= lid < n_labels):
        return None, None
    return lid, float(np.clip(conf, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="test examples to evaluate")
    ap.add_argument("--model", default=config.FRONTIER_MODEL)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env).")

    from openai import OpenAI

    labels = json.loads((config.PROC_DIR / "label_names.json").read_text("utf-8"))
    df = pd.read_parquet(config.PROC_DIR / f"{args.split}.parquet")
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=args.seed).reset_index(drop=True)

    client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)

    y_true, y_pred, conf = [], [], []
    latencies, total_cost, parse_fail, records = [], 0.0, 0, []

    from tqdm import tqdm

    for _, row in tqdm(df.iterrows(), total=len(df), desc=args.model):
        msgs = build_messages(row["text"], labels)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=msgs,
                temperature=0,
                max_tokens=40,
                response_format={"type": "json_object"},
                extra_body={"usage": {"include": True}},  # ask OpenRouter for $ cost
            )
        except Exception as e:  # retry once on transient errors
            time.sleep(2)
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=msgs, temperature=0, max_tokens=40
                )
            except Exception:
                parse_fail += 1
                records.append({"gold": int(row["label"]), "error": str(e)[:200]})
                continue
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        # cost (OpenRouter returns usage.cost in USD when include=true)
        try:
            total_cost += float(getattr(resp.usage, "cost", 0.0) or 0.0)
        except Exception:
            pass

        lid, c = parse_reply(resp.choices[0].message.content, len(labels))
        if lid is None:
            parse_fail += 1
            records.append({"gold": int(row["label"]),
                            "raw": (resp.choices[0].message.content or "")[:200]})
            continue
        y_true.append(int(row["label"]))
        y_pred.append(lid)
        conf.append(c)
        records.append({"gold": int(row["label"]), "pred": lid, "conf": c})

    if not y_true:
        raise SystemExit("No parseable predictions — check model/output format.")

    res, bins = evaluate(np.array(y_true), np.array(y_pred), np.array(conf))
    summary = {
        "model": args.model,
        "n_evaluated": len(y_true),
        "n_parse_failures": parse_fail,
        "metrics": res.as_dict(),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "total_cost_usd": total_cost,
        "cost_per_1k_usd": (total_cost / len(y_true) * 1000) if y_true else None,
        "reliability_bins": bins,
    }
    slug = args.model.replace("/", "_")
    out = config.OUTPUT_DIR / f"frontier_{slug}.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2), "utf-8")

    print("\n=== Frontier baseline:", args.model, "===")
    for k, v in res.as_dict().items():
        print(f"  {k:10}= {v:.4f}" if isinstance(v, float) else f"  {k:10}= {v}")
    print(f"  latency_ms= {summary['latency_ms_mean']:.1f} (mean)")
    print(f"  cost/1k   = ${summary['cost_per_1k_usd']:.4f}")
    print(f"  parse_fail= {parse_fail}")
    print("Wrote", out)


if __name__ == "__main__":
    main()
