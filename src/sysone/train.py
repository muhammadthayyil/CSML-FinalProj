"""System One reproduction: frozen Qwen backbone + LoRA + a decision head that scores
ALL product options in a single forward pass (non-autoregressive), then calibrated.

This follows the open "decision model" recipe (frozen LM weights + rank-32 LoRA adapters
+ a decision head, trained jointly): see "Releasing the Inner Jev" (arXiv 2610.17205).

Implementation: a decoder LM with a sequence-classification head. The base model is frozen;
only the LoRA adapters and the K-way decision head train. One forward pass yields a
probability over the fixed option set — the System One "typed decision" contract.

Run on the RENTED GPU (needs CUDA; too big for the 4GB laptop):
    pip install torch transformers datasets accelerate peft scikit-learn scipy pandas
    # optional 4-bit (bigger backbones):  pip install bitsandbytes
    python -m src.sysone.train --base Qwen/Qwen2.5-0.5B --epochs 3 --bs 16
    python -m src.sysone.train --base Qwen/Qwen2.5-1.5B --epochs 3 --bs 8 --load-4bit

Outputs:
    checkpoints/sysone_<slug>/         LoRA adapter + head + tokenizer
    outputs/sysone_<slug>.json         metrics (uncalibrated & temperature-calibrated),
                                       fitted temperature, reliability bins, latency
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.metrics import evaluate, expected_calibration_error, temperature_scale


def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def build_compute_metrics():
    """Trainer compute_metrics: logits -> accuracy, macro-F1, ECE."""
    def compute(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits, dtype=np.float64)
        probs = _softmax(logits, axis=1)
        pred = probs.argmax(1)
        conf = probs.max(1)
        res, _ = evaluate(labels, pred, conf, probs=probs)
        return {"accuracy": res.accuracy, "macro_f1": res.macro_f1, "ece": res.ece}
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SYS1_BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16, help="per-device batch size")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--train-size", type=int, default=0, help="subsample train (0=all)")
    ap.add_argument("--report", default="tensorboard", choices=["none", "tensorboard", "wandb"],
                    help="live monitoring backend")
    ap.add_argument("--eval-steps", type=int, default=200, help="eval/checkpoint every N steps")
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (evals)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="save VRAM (use for 7B)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(f"Missing ML deps ({e}). Install on the GPU box — see the module docstring.")

    labels = json.loads((config.PROC_DIR / "label_names.json").read_text("utf-8"))
    num_labels = len(labels)
    slug = args.base.split("/")[-1].lower()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # bf16 on A40/L4, fp16 on T4

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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        d = Dataset.from_pandas(df, preserve_index=False)
        d = d.map(tokenize, batched=True, remove_columns=["text"])
        ds[name] = d

    # ---- model: frozen backbone + LoRA + decision head -----------------------
    model_kwargs = dict(num_labels=num_labels, low_cpu_mem_usage=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(args.base, **model_kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,          # keeps the "score" decision head trainable
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if args.grad_checkpoint:
        model.config.use_cache = False
        model.enable_input_require_grads()   # required for gradient checkpointing with PEFT

    # ---- train ---------------------------------------------------------------
    ckpt_dir = config.CKPT_DIR / f"sysone_{slug}"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    # ---- calibrate on val, evaluate on test ----------------------------------
    val_logits = trainer.predict(ds["val"]).predictions
    val_labels = np.array(val_df["labels"])
    T = temperature_scale(np.asarray(val_logits, dtype=np.float64), val_labels)
    print(f"Fitted temperature T = {T:.3f}")

    test_out = trainer.predict(ds["test"])
    test_logits = np.asarray(test_out.predictions, dtype=np.float64)
    test_labels = np.array(test_df["labels"])

    def metrics_at(logits):
        probs = _softmax(logits, axis=1)
        pred, conf = probs.argmax(1), probs.max(1)
        res, bins = evaluate(test_labels, pred, conf, probs=probs)
        return res, bins

    res_u, bins_u = metrics_at(test_logits)          # uncalibrated
    res_c, bins_c = metrics_at(test_logits / T)      # temperature-calibrated

    # ---- latency (ms/decision, batch=1) on a small sample --------------------
    model.eval()
    dev = next(model.parameters()).device
    sample = test_df["text"].sample(min(100, len(test_df)), random_state=args.seed).tolist()
    lat = []
    with torch.no_grad():
        for t in sample:
            enc = tok(t, truncation=True, max_length=args.max_length, return_tensors="pt").to(dev)
            torch.cuda.synchronize() if dev.type == "cuda" else None
            t0 = time.perf_counter()
            model(**enc)
            torch.cuda.synchronize() if dev.type == "cuda" else None
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
    out = config.OUTPUT_DIR / f"sysone_{slug}.json"
    out.write_text(json.dumps(summary, indent=2), "utf-8")

    print("\n=== System One repro:", args.base, "===")
    print("  uncalibrated:", {k: round(v, 4) for k, v in res_u.as_dict().items()})
    print("  calibrated  :", {k: round(v, 4) for k, v in res_c.as_dict().items()})
    print(f"  ECE {res_u.ece:.4f} -> {res_c.ece:.4f} after T={T:.3f}")
    print(f"  latency {summary['latency_ms_mean']:.1f} ms/decision")
    print("Wrote", out, "and", ckpt_dir)


if __name__ == "__main__":
    main()
