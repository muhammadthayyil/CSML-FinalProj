"""Download the CFPB snapshot and build clean, stratified train/val/test splits.

Usage (run on any machine with internet + `datasets` installed):
    python -m src.data.prepare_data                 # defaults from config.py
    python -m src.data.prepare_data --max-per-class 5000 --seed 42

Output (data/processed/):
    train.parquet, val.parquet, test.parquet   columns: [text, label, label_name]
    label_names.json                            the ordered class list

Why a HF snapshot and not the CFPB API: the CFPB has removed consumer narratives
from every current public feed (API / CSV export / bulk zip). The snapshot
`milesbutler/consumer_complaints` was captured while narratives were public and keeps
both the Product label and the narrative text. See src/config.py for details.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src import config


def _load_raw() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Please `pip install datasets pandas pyarrow` first.")

    print(f"[1/5] Loading {config.HF_DATASET} from Hugging Face ...")
    ds = load_dataset(config.HF_DATASET, split="train")
    # Keep only the two columns we need (saves a lot of memory).
    keep = [c for c in (config.PRODUCT_COL, config.TEXT_COL) if c in ds.column_names]
    if len(keep) < 2:
        sys.exit(f"Expected columns {config.PRODUCT_COL!r} and {config.TEXT_COL!r}; "
                 f"found {ds.column_names}")
    ds = ds.select_columns(keep)
    df = ds.to_pandas()
    df = df.rename(columns={config.PRODUCT_COL: "product_raw", config.TEXT_COL: "text"})
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[2/5] Cleaning {len(df):,} raw rows ...")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= config.MIN_NARRATIVE_CHARS]
    # Map raw product -> consolidated canonical label; drop anything unmapped.
    df["label_name"] = df["product_raw"].map(config.PRODUCT_CONSOLIDATION)
    df = df.dropna(subset=["label_name"])
    df["label"] = df["label_name"].map(config.LABEL2ID).astype(int)
    df = df.drop_duplicates(subset=["text"])
    return df[["text", "label", "label_name"]].reset_index(drop=True)


def _balance(df: pd.DataFrame, max_per_class, seed: int) -> pd.DataFrame:
    if not max_per_class:
        return df
    print(f"[3/5] Capping each class at {max_per_class:,} ...")
    parts = [
        g.sample(min(len(g), max_per_class), random_state=seed)
        for _, g in df.groupby("label_name")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _split(df: pd.DataFrame, seed: int):
    from sklearn.model_selection import train_test_split

    print("[4/5] Stratified split into train/val/test ...")
    test_frac = config.TEST_FRAC
    val_frac = config.VAL_FRAC
    train, tmp = train_test_split(
        df, test_size=test_frac + val_frac, stratify=df["label"], random_state=seed
    )
    rel = test_frac / (test_frac + val_frac)
    val, test = train_test_split(
        tmp, test_size=rel, stratify=tmp["label"], random_state=seed
    )
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=config.MAX_PER_CLASS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    df = _clean(_load_raw())
    df = _balance(df, args.max_per_class, args.seed)
    train, val, test = _split(df, args.seed)

    print("[5/5] Writing parquet to", config.PROC_DIR)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_parquet(config.PROC_DIR / f"{name}.parquet")
    (config.PROC_DIR / "label_names.json").write_text(
        json.dumps(config.LABELS, indent=2), encoding="utf-8"
    )

    print("\n=== Class distribution (full, pre-split) ===")
    print(df["label_name"].value_counts().to_string())
    print(f"\nTotals  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print("Done.")


if __name__ == "__main__":
    main()
