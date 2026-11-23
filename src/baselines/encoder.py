"""Encoder baseline: a fully fine-tuned DeBERTa classifier — a strong, cheap, single-pass
comparison point for the System One reproduction. Same typed-decision task (narrative ->
product), same metrics (accuracy / macro-F1 / ECE / Brier / latency), same temperature
calibration, so results drop straight into the comparison table.

Small enough to run on a modest GPU (and slowly on CPU for a smoke test).

Run:
    pip install torch transformers datasets accelerate scikit-learn scipy pandas sentencepiece
    python -m src.baselines.encoder --base microsoft/deberta-v3-small --epochs 3 --bs 16

Outputs:
    checkpoints/encoder_<slug>/      fine-tuned model + tokenizer
    outputs/encoder_<slug>.json      metrics (uncalibrated & calibrated), temperature, latency
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.metrics import evaluate, temperature_scale


def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.ENCODER_BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--train-size", type=int, default=0, help="subsample train (0=all)")
    ap.add_argument("--report", default="tensorboard", choices=["none", "tensorboard", "wandb"])
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (evals)")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as e:
        raise SystemExit(f"Missing ML deps ({e}). See the module docstring for the pip line.")

    labels = json.loads((config.PROC_DIR / "label_names.json").read_text("utf-8"))
    num_labels = len(labels)
    slug = args.base.split("/")[-1].lower()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # ---- data ----------------------------------------------------------------
    def load_split(name):
        df = pd.read_parquet(config.PROC_DIR / f"{name}.parquet")[["text", "label"]]
        return df.rename(columns={"label": "labels"})

    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    if args.train_size:
        train_df = train_df.sample(min(args.train_size, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
    print(f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} K={num_labels}")

    tok = AutoTokenizer.from_pretrained(args.base)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        d = Dataset.from_pandas(df, preserve_index=False)
        ds[name] = d.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=num_labels)

    def compute_metrics(eval_pred):
        logits, lbls = eval_pred
        probs = _softmax(np.asarray(logits, dtype=np.float64), axis=1)
        res, _ = evaluate(lbls, probs.argmax(1), probs.max(1), probs=probs)
        return {"accuracy": res.accuracy, "macro_f1": res.macro_f1, "ece": res.ece}

    # ---- train ---------------------------------------------------------------
    ckpt_dir = config.CKPT_DIR / f"encoder_{slug}"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(args.bs, 32),
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=args.grad_checkpoint,
        report_to=args.report,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    # Auto-resume: if a rerun finds a valid checkpoint in output_dir, continue from it;
    # otherwise None = fresh start.
    last_ckpt = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if last_ckpt:
        print(f"Found checkpoint — resuming from {last_ckpt}")
    else:
        print("No checkpoint found — training from scratch")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # ---- calibrate on val, evaluate on test ----------------------------------
    val_logits = np.asarray(trainer.predict(ds["val"]).predictions, dtype=np.float64)
    T = temperature_scale(val_logits, np.array(val_df["labels"]))
    print(f"Fitted temperature T = {T:.3f}")

    test_logits = np.asarray(trainer.predict(ds["test"]).predictions, dtype=np.float64)
    test_labels = np.array(test_df["labels"])

    def metrics_at(logits):
        probs = _softmax(logits, axis=1)
        res, bins = evaluate(test_labels, probs.argmax(1), probs.max(1), probs=probs)
        return res, bins

    res_u, _ = metrics_at(test_logits)
    res_c, bins_c = metrics_at(test_logits / T)

    # ---- latency (ms/decision, batch=1) --------------------------------------
    model.eval()
    dev = next(model.parameters()).device
    sample = test_df["text"].sample(min(100, len(test_df)), random_state=args.seed).tolist()
    lat = []
    with torch.no_grad():
        for t in sample:
            enc = tok(t, truncation=True, max_length=args.max_length, return_tensors="pt").to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)

    # ---- save ----------------------------------------------------------------
    trainer.save_model(str(ckpt_dir))
    tok.save_pretrained(str(ckpt_dir))
    summary = {
        "base_model": args.base,
        "num_labels": num_labels,
        "temperature": T,
        "metrics_uncalibrated": res_u.as_dict(),
        "metrics_calibrated": res_c.as_dict(),
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "reliability_bins_calibrated": bins_c,
        "args": vars(args),
    }
    out = config.OUTPUT_DIR / f"encoder_{slug}.json"
    out.write_text(json.dumps(summary, indent=2), "utf-8")

    print("\n=== Encoder baseline:", args.base, "===")
    print("  uncalibrated:", {k: round(v, 4) for k, v in res_u.as_dict().items()})
    print("  calibrated  :", {k: round(v, 4) for k, v in res_c.as_dict().items()})
    print(f"  ECE {res_u.ece:.4f} -> {res_c.ece:.4f} after T={T:.3f}")
    print(f"  latency {summary['latency_ms_mean']:.1f} ms/decision")
    print("Wrote", out, "and", ckpt_dir)


if __name__ == "__main__":
    main()
