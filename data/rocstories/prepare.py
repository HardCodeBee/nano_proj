"""
Convert ROCStories into nanoGPT-ready local artifacts.

Outputs:
- train.bin / val.bin: uint16 GPT-2 token streams for nanoGPT training
- dataset_stats.json: split-level token-length statistics
- test_full.txt: blank-line-separated validation stories for exact paragraph eval

Processing:
Each story is tokenized with the GPT-2 BPE tokenizer, then an end-of-text token is
appended so the model can know where one story stops and the next one begins.
"""

import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


DATASET_ID = "mintujupally/ROCStories"
TEXT_KEY = "text"
NUM_PROC = 8
# Reuse the same tokenizer family 
ENC = tiktoken.get_encoding("gpt2")


def process(example):
    # Encode one full story and add an explicit boundary marker between stories.
    # enc.encode_ordinary ignores any special tokens
    ids = ENC.encode_ordinary(example[TEXT_KEY])

    ids.append(ENC.eot_token)
    return {"ids": ids, "len": len(ids)}

# Summarize story length statistics for a given split.
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
    # eval.py expects paragraphs separated by blank lines in the .txt mode.
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(story.strip() for story in stories if story.strip()))
        f.write("\n")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    dataset = load_dataset(DATASET_ID)
    dataset = {
        #using the public test split for local evaluation/model selection.
        "train": dataset["train"],
        "val": dataset["test"],
    }

    print("Loaded ROCStories dataset:")
    for split, dset in dataset.items():
        print(f"  {split}: {len(dset):,} stories")

    tokenized = {}
    stats = {
        "dataset_id": DATASET_ID,
        "tokenizer": "gpt2",
        "separator_token_id": int(ENC.eot_token),
        "splits": {},
    }

    for split, dset in dataset.items():
        tokenized_split = dset.map(
            process,
            remove_columns=dset.column_names,
            desc=f"tokenizing {split}",
            num_proc=NUM_PROC,
        )
        tokenized[split] = tokenized_split

        split_stats = summarize_lengths(tokenized_split["len"])
        split_stats["stories"] = len(tokenized_split)
        split_stats["tokens_total"] = int(np.sum(tokenized_split["len"], dtype=np.uint64))
        stats["splits"][split] = split_stats

        filename = out_dir / f"{split}.bin"
        arr_len = split_stats["tokens_total"]
        # nanoGPT reads one long uint16 token stream via np.memmap during training.
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        total_batches = 256
        for batch_idx in range(total_batches):
            batch = tokenized_split.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")
            # Writing split shards avoids materializing the whole tokenized dataset at once.
            arr_batch = np.concatenate(batch["ids"]) if len(batch) else np.array([], dtype=np.uint16)
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(
            f"{split}.bin: {split_stats['tokens_total']:,} tokens, "
            f"mean/story={split_stats['mean']:.2f}, p95={split_stats['p95']}"
        )

    eval_text_path = out_dir / "test_full.txt"
    write_eval_text(eval_text_path, dataset["val"][TEXT_KEY])
    print(f"Saved evaluation text to {eval_text_path}")

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
