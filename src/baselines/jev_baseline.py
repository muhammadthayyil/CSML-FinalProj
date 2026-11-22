"""Jev reference baseline: benchmark the REAL System One model (TypeSafe Jev, via
OpenRouter's Decisions API) on the same test set, with the same metrics, so Jev anchors
the comparison table as the gold reference our open reproduction is trying to approximate.

Why this is NOT the chat endpoint: Jev is a *structured decision* model. OpenRouter serves
it through the **Decisions API** (`POST /api/alpha/decisions`), not chat completions. We
send a `state` (the complaint) and a typed `choice` question whose `criteria` are the
product categories; Jev returns the chosen option AND a calibrated probability distribution
over the options — exactly what we need for accuracy + ECE + Brier.

Note: Jev cannot be fine-tuned (closed, hosted). It is the *reference*, not a model we
train. Our contribution remains the open Qwen+LoRA reproduction (src/sysone/train.py).

Run (on a network that can reach OpenRouter — e.g. the rented box, NOT the filtered laptop):
    pip install requests python-dotenv pandas scikit-learn scipy numpy tqdm
    python -m src.baselines.jev_baseline --probe            # 1 call, prints raw JSON to verify shape
    python -m src.baselines.jev_baseline --n 300 --workers 8

Reads OPENROUTER_API_KEY from .env. Writes outputs/jev_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from src import config          # noqa: E402
from src.metrics import evaluate  # noqa: E402

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000  # $0.042 / 1M input tokens (output free)


def build_request(text: str, model: str, labels: list[str]) -> dict:
    """One typed `choice` question over the product categories.
    Option keys are c0..c{K-1}; descriptions are the human label names."""
    criteria = {f"c{i}": name for i, name in enumerate(labels)}
    return {
        "model": model,
        "state": text[:6000],
        "questions": {
            "product": {
                "type": "choice",
                "instructions": (
                    "Which U.S. consumer-finance product category does this complaint "
                    "concern? Choose the single best-matching category."
                ),
                "criteria": criteria,
            }
        },
    }


def parse_answer(resp_json: dict, n_labels: int):
    """Return (pred_idx, confidence, prob_vec[np K], prompt_tokens) or (None,...)."""
    ans = resp_json.get("answers") or resp_json.get("decision", {}).get("answers") or {}
    p = ans.get("product", {})
    choice_key = p.get("choice")
    probs_map = p.get("probabilities") or {}
    if choice_key is None:
        return None, None, None, 0

    vec = np.zeros(n_labels, dtype=float)
    for k, v in probs_map.items():
        if k.startswith("c") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < n_labels:
                vec[idx] = float(v)
    s = vec.sum()
    if s > 0:
        vec = vec / s                      # renormalize the returned distribution
        pred = int(vec.argmax())
        conf = float(vec[pred])
    else:                                  # no distribution returned: fall back to the choice
        pred = int(choice_key[1:]) if choice_key[1:].isdigit() else None
        if pred is None:
            return None, None, None, 0
        vec[pred] = 1.0
        conf = 1.0

    usage = resp_json.get("usage") or {}
    ptoks = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    return pred, conf, vec, ptoks


def call_jev(session: requests.Session, text: str, model: str, labels: list[str], headers: dict):
    body = build_request(text, model, labels)
    t0 = time.perf_counter()
    r = session.post(DECISIONS_URL, headers=headers, json=body, timeout=60)
    dt = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="test examples (0 = all)")
    ap.add_argument("--model", default="typesafe/jev-1.14")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--probe", action="store_true", help="make ONE call and print raw JSON")
    args = ap.parse_args()

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env).")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    labels = json.loads((config.PROC_DIR / "label_names.json").read_text("utf-8"))
    df = pd.read_parquet(config.PROC_DIR / f"{args.split}.parquet")

    session = requests.Session()

    # Probe: verify the live response shape before a full run (recommended first step).
    if args.probe:
        raw, dt = call_jev(session, df.iloc[0]["text"], args.model, labels, headers)
        print(f"latency={dt:.0f}ms\nRAW RESPONSE:\n" + json.dumps(raw, indent=2)[:3000])
        return

    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=args.seed).reset_index(drop=True)

    def work(row):
        try:
            raw, dt = call_jev(session, row["text"], args.model, labels, headers)
            pred, conf, vec, ptoks = parse_answer(raw, len(labels))
            if pred is None:
                return {"ok": False, "gold": int(row["label"])}
            return {"ok": True, "gold": int(row["label"]), "pred": pred,
                    "conf": conf, "vec": vec, "lat": dt, "ptoks": ptoks}
        except Exception as e:
            return {"ok": False, "gold": int(row["label"]), "err": str(e)[:200]}

    from tqdm import tqdm

    rows = [r for _, r in df.iterrows()]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for res in tqdm(ex.map(work, rows), total=len(rows), desc=args.model):
            results.append(res)

    ok = [r for r in results if r.get("ok")]
    if not ok:
        raise SystemExit("No successful Jev decisions — run with --probe to inspect the response.")

    y_true = np.array([r["gold"] for r in ok])
    y_pred = np.array([r["pred"] for r in ok])
    conf = np.array([r["conf"] for r in ok])
    probs = np.vstack([r["vec"] for r in ok])
    latencies = [r["lat"] for r in ok]
    total_cost = sum(r["ptoks"] for r in ok) * INPUT_PRICE_PER_TOKEN

    res, bins = evaluate(y_true, y_pred, conf, probs=probs)
    summary = {
        "model": args.model,
        "n_evaluated": len(ok),
        "n_failures": len(results) - len(ok),
        "metrics": res.as_dict(),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "total_cost_usd": total_cost,
        "cost_per_1k_usd": total_cost / len(ok) * 1000,
        "reliability_bins": bins,
    }
    slug = args.model.replace("/", "_")
    out = config.OUTPUT_DIR / f"jev_{slug}.json"
    out.write_text(json.dumps(summary, indent=2), "utf-8")

    print("\n=== Jev reference:", args.model, "===")
    for k, v in res.as_dict().items():
        print(f"  {k:10}= {v:.4f}" if isinstance(v, float) else f"  {k:10}= {v}")
    print(f"  latency_ms= {summary['latency_ms_mean']:.1f} (mean)  p50={summary['latency_ms_p50']:.1f}")
    print(f"  cost/1k   = ${summary['cost_per_1k_usd']:.4f}")
    print(f"  failures  = {summary['n_failures']}")
    print("Wrote", out)


if __name__ == "__main__":
    main()
