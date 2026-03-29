"""
Prepare TinyStories for Task 2 narrative-curriculum experiments.

Outputs:
- train.bin / val.bin: uint16 GPT-2 token streams for nanoGPT training
- dataset_stats.json: split-level token-length statistics and preparation args
- val_full.txt: blank-line-separated validation stories for manual inspection

Design goal:
Keep preprocessing aligned with ROCStories so Stage 1 -> Stage 2 curriculum
experiments do not introduce a tokenizer or separator mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


DATASET_ID = "roneneldan/TinyStories"
TEXT_KEY = "text"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
NUM_PROC = 8
TOTAL_BATCHES = 256
ENC = tiktoken.get_encoding("gpt2")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TinyStories with GPT-2 BPE + EOT.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--train-split", default=TRAIN_SPLIT)
    parser.add_argument("--val-split", default=VAL_SPLIT)
    parser.add_argument("--text-key", default=TEXT_KEY)
    parser.add_argument("--num-proc", type=int, default=NUM_PROC)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional cap for a controllable Stage 1 subset.",
    )
    parser.add_argument(
        "--max-val-examples",
        type=int,
        default=None,
        help="Optional cap for validation stories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Used only when shuffling before subset selection.",
    )
    return parser.parse_args()


def process(example, text_key):
    ids = ENC.encode_ordinary(example[text_key])
    ids.append(ENC.eot_token)
    return {"ids": ids, "len": len(ids)}


def summarize_lengths(lengths):
    arr = np.array(lengths, dtype=np.int64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def maybe_select_subset(dset, max_examples, seed):
    if max_examples is None or max_examples <= 0 or len(dset) <= max_examples:
        return dset, False
    shuffled = dset.shuffle(seed=seed)
    return shuffled.select(range(max_examples)), True


def write_eval_text(path, stories):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(story.strip() for story in stories if story.strip()))
        handle.write("\n")


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(__file__).resolve().parent

    dataset = load_dataset(args.dataset_id)
    split_map = {
        "train": dataset[args.train_split],
        "val": dataset[args.val_split],
    }

    train_selected, train_was_subset = maybe_select_subset(
        split_map["train"],
        args.max_train_examples,
        args.seed,
    )
    val_selected, val_was_subset = maybe_select_subset(
        split_map["val"],
        args.max_val_examples,
        args.seed,
    )
    split_map = {"train": train_selected, "val": val_selected}

    print("Loaded TinyStories dataset:")
    for split, dset in split_map.items():
        subset_note = ""
        if split == "train" and train_was_subset:
            subset_note = " (subset)"
        if split == "val" and val_was_subset:
            subset_note = " (subset)"
        print(f"  {split}: {len(dset):,} stories{subset_note}")

    tokenized = {}
    stats = {
        "dataset_id": args.dataset_id,
        "tokenizer": "gpt2",
        "separator_token_id": int(ENC.eot_token),
        "args": {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "max_train_examples": args.max_train_examples,
            "max_val_examples": args.max_val_examples,
            "seed": args.seed,
        },
        "splits": {},
    }

    for split, dset in split_map.items():
        tokenized_split = dset.map(
            process,
            fn_kwargs={"text_key": args.text_key},
            remove_columns=dset.column_names,
            desc=f"tokenizing {split}",
            num_proc=args.num_proc,
        )
        tokenized[split] = tokenized_split

        split_stats = summarize_lengths(tokenized_split["len"])
        split_stats["stories"] = len(tokenized_split)
        split_stats["tokens_total"] = int(np.sum(tokenized_split["len"], dtype=np.uint64))
        stats["splits"][split] = split_stats

        filename = out_dir / f"{split}.bin"
        arr_len = split_stats["tokens_total"]
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        for batch_idx in range(TOTAL_BATCHES):
            batch = tokenized_split.shard(
                num_shards=TOTAL_BATCHES,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"]) if len(batch) else np.array([], dtype=np.uint16)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(
            f"{split}.bin: {split_stats['tokens_total']:,} tokens, "
            f"mean/story={split_stats['mean']:.2f}, p95={split_stats['p95']}"
        )

    eval_text_path = out_dir / "val_full.txt"
    write_eval_text(eval_text_path, split_map["val"][args.text_key])
    print(f"Saved validation text to {eval_text_path}")

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Saved stats to {stats_path}")
