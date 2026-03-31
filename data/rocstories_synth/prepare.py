"""
Prepare locally generated ROCStories-style synthetic stories for nanoGPT.

Expected input: a JSONL file where each accepted row has a `story` field.
The output mirrors the ROCStories local protocol:

- train.bin
- val.bin
- train_story_starts.npy / train_story_lengths.npy
- train_first_sentence_lengths.npy
- val_story_starts.npy / val_story_lengths.npy
- val_first_sentence_lengths.npy
- val_full.txt
- dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tiktoken


ENC = tiktoken.get_encoding("gpt2")
DEFAULT_INPUT = Path(__file__).resolve().parent / "raw" / "accepted.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare synthetic ROCStories-style data.")
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--dataset-label", default="synthetic_rocstories_style")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


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


def encode_story(story: str) -> list[int]:
    ids = ENC.encode_ordinary(story)
    ids.append(ENC.eot_token)
    return ids


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.replace("\n", " ").split()).strip()
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def first_sentence_token_length(story: str) -> int:
    sentences = split_sentences(story)
    first_sentence = sentences[0] if sentences else " ".join(story.split()).strip()
    return len(ENC.encode_ordinary(first_sentence)) if first_sentence else 0


def write_eval_text(path: Path, stories: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n\n".join(story.strip() for story in stories if story.strip()))
        handle.write("\n")


def write_split(out_dir: Path, split_name: str, stories: list[str], stats: dict) -> None:
    encoded = [encode_story(story) for story in stories]
    lengths = np.array([len(ids) for ids in encoded], dtype=np.int64)
    starts = np.cumsum(np.concatenate(([0], lengths[:-1])), dtype=np.int64) if len(lengths) else np.array([], dtype=np.int64)
    first_sentence_lengths = np.array([first_sentence_token_length(story) for story in stories], dtype=np.int64)
    total_tokens = int(np.sum(lengths, dtype=np.uint64))

    stats["splits"][split_name] = summarize_lengths(lengths)
    stats["splits"][split_name]["stories"] = len(stories)
    stats["splits"][split_name]["tokens_total"] = total_tokens

    bin_path = out_dir / f"{split_name}.bin"
    arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    idx = 0
    for ids in encoded:
        ids_arr = np.array(ids, dtype=np.uint16)
        arr[idx : idx + len(ids_arr)] = ids_arr
        idx += len(ids_arr)
    arr.flush()

    np.save(out_dir / f"{split_name}_story_starts.npy", starts)
    np.save(out_dir / f"{split_name}_story_lengths.npy", lengths)
    np.save(out_dir / f"{split_name}_first_sentence_lengths.npy", first_sentence_lengths)

    print(
        f"{split_name}.bin: {total_tokens:,} tokens, "
        f"mean/story={stats['splits'][split_name]['mean']:.2f}, "
        f"p95={stats['splits'][split_name]['p95']}"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing synthetic input file: {input_path}")

    stories = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            story = " ".join(str(record.get("story", "")).split()).strip()
            if story:
                stories.append(story)

    if len(stories) < 100:
        raise ValueError("Synthetic dataset is too small. Generate more accepted stories before preparing.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(stories))
    rng.shuffle(indices)
    stories = [stories[idx] for idx in indices]

    val_count = max(1, int(round(len(stories) * args.val_fraction)))
    val_stories = stories[:val_count]
    train_stories = stories[val_count:]

    stats = {
        "dataset_id": args.dataset_label,
        "source_jsonl": str(input_path),
        "tokenizer": "gpt2",
        "separator_token_id": int(ENC.eot_token),
        "split_policy": {
            "train_source": "accepted_synthetic_jsonl",
            "val_source": "accepted_synthetic_jsonl",
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "splits": {},
    }

    print("Loaded synthetic ROCStories-style dataset:")
    print(f"  train: {len(train_stories):,} stories")
    print(f"  val: {len(val_stories):,} stories")

    write_split(out_dir, "train", train_stories, stats)
    write_split(out_dir, "val", val_stories, stats)

    val_text_path = out_dir / "val_full.txt"
    write_eval_text(val_text_path, val_stories)
    print(f"Saved validation text to {val_text_path}")

    stats_path = out_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
