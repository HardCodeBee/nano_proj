"""
Convert ROCStories into nanoGPT-ready local artifacts with a three-layer split:

- train.bin: official train minus a small held-out validation slice
- val.bin: held-out slice from the official train split, used for checkpoint selection
- train_story_starts.npy / train_story_lengths.npy: story boundary metadata for story-aware sampling
- train_first_sentence_lengths.npy: token lengths of the opening sentence for continuation-aware weighting
- val_story_starts.npy / val_story_lengths.npy: validation metadata in the same format
- val_first_sentence_lengths.npy: validation opening-sentence token lengths
- val_full.txt: blank-line-separated validation stories for day-to-day paragraph eval
- locked_test.txt: the untouched official public test split, reserved for occasional final checks
- dataset_stats.json: split-level token-length statistics and split metadata

This keeps Task 2 model selection off the public test split while preserving the
official test stories as a locked local benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


DATASET_ID = "mintujupally/ROCStories"
TEXT_KEY = "text"
NUM_PROC = 8
VAL_FRACTION = 0.05
SPLIT_SEED = 2027
ENC = tiktoken.get_encoding("gpt2")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ROCStories for Task 2.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--text-key", default=TEXT_KEY)
    parser.add_argument("--num-proc", type=int, default=NUM_PROC)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="Fraction of the official train split to hold out as validation.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def process(example, text_key):
    text = example[text_key]
    ids = ENC.encode_ordinary(text)
    ids.append(ENC.eot_token)
    first_sentence = extract_first_sentence(text)
    first_sentence_len = len(ENC.encode_ordinary(first_sentence)) if first_sentence else 0
    return {"ids": ids, "len": len(ids), "first_sentence_len": first_sentence_len}


def split_sentences(text):
    normalized = " ".join(str(text).replace("\n", " ").split()).strip()
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def extract_first_sentence(text):
    sentences = split_sentences(text)
    return sentences[0] if sentences else " ".join(str(text).split()).strip()


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


def write_eval_text(path, stories):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(story.strip() for story in stories if story.strip()))
        handle.write("\n")


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(__file__).resolve().parent

    dataset = load_dataset(args.dataset_id)
    train_split = dataset["train"].train_test_split(
        test_size=args.val_fraction,
        seed=args.seed,
        shuffle=True,
    )
    split_map = {
        "train": train_split["train"],
        "val": train_split["test"],
    }
    locked_test = dataset["test"]

    print("Loaded ROCStories dataset:")
    print(f"  train: {len(split_map['train']):,} stories")
    print(f"  val: {len(split_map['val']):,} stories (held out from official train)")
    print(f"  locked_test: {len(locked_test):,} stories (official public test)")

    tokenized = {}
    stats = {
        "dataset_id": args.dataset_id,
        "tokenizer": "gpt2",
        "separator_token_id": int(ENC.eot_token),
        "split_policy": {
            "train_source": "official_train",
            "val_source": "official_train",
            "val_fraction": args.val_fraction,
            "locked_test_source": "official_test",
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
        lengths = np.array(tokenized_split["len"], dtype=np.int64)
        starts = np.cumsum(np.concatenate(([0], lengths[:-1])), dtype=np.int64)
        first_sentence_lengths = np.array(tokenized_split["first_sentence_len"], dtype=np.int64)

        filename = out_dir / f"{split}.bin"
        arr_len = split_stats["tokens_total"]
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        total_batches = 256
        for batch_idx in range(total_batches):
            batch = tokenized_split.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"]) if len(batch) else np.array([], dtype=np.uint16)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        starts_path = out_dir / f"{split}_story_starts.npy"
        lengths_path = out_dir / f"{split}_story_lengths.npy"
        first_sentence_lengths_path = out_dir / f"{split}_first_sentence_lengths.npy"
        np.save(starts_path, starts)
        np.save(lengths_path, lengths)
        np.save(first_sentence_lengths_path, first_sentence_lengths)

        print(
            f"{split}.bin: {split_stats['tokens_total']:,} tokens, "
            f"mean/story={split_stats['mean']:.2f}, p95={split_stats['p95']}"
        )
        print(
            "Saved story metadata to "
            f"{starts_path.name}, {lengths_path.name}, and {first_sentence_lengths_path.name}"
        )

    locked_test_tokenized = locked_test.map(
        process,
        fn_kwargs={"text_key": args.text_key},
        remove_columns=locked_test.column_names,
        desc="tokenizing locked_test for stats",
        num_proc=args.num_proc,
    )
    locked_stats = summarize_lengths(locked_test_tokenized["len"])
    locked_stats["stories"] = len(locked_test_tokenized)
    locked_stats["tokens_total"] = int(np.sum(locked_test_tokenized["len"], dtype=np.uint64))
    stats["splits"]["locked_test"] = locked_stats

    val_text_path = out_dir / "val_full.txt"
    write_eval_text(val_text_path, split_map["val"][args.text_key])
    print(f"Saved validation text to {val_text_path}")

    locked_test_path = out_dir / "locked_test.txt"
    write_eval_text(locked_test_path, locked_test[args.text_key])
    print(f"Saved locked test text to {locked_test_path}")

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Saved stats to {stats_path}")
