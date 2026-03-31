"""
Summarize Task 2 sample JSONL files with a few lightweight continuation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TERMINAL_PUNCTUATION = (".", "!", "?", '"', "'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task 2 sample JSONL outputs.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <input>.summary.json or .csv depending on --format.",
    )
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    return parser.parse_args()


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(str(text).replace("\n", " ").split()).strip()
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def prompt_to_first_sentence_overlap(prompt: str, continuation_text: str) -> float:
    prompt_tokens = set(word_tokens(prompt))
    if not prompt_tokens:
        return 0.0
    continuation_sentences = split_sentences(continuation_text)
    if not continuation_sentences:
        return 0.0
    first_sentence_tokens = set(word_tokens(continuation_sentences[0]))
    return len(prompt_tokens & first_sentence_tokens) / len(prompt_tokens)


def distinct_4_ratio(text: str) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    if len(tokens) < 4:
        return 1.0
    ngrams = [tuple(tokens[idx : idx + 4]) for idx in range(len(tokens) - 3)]
    return len(set(ngrams)) / len(ngrams)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No sample records found in {path}")
    return records


def summarize_records(records: list[dict], input_path: Path) -> dict:
    overlap_values = []
    sentence_counts = []
    ended_with_eot_flags = []
    terminal_punctuation_flags = []
    distinct_4_values = []
    generated_tokens = []

    for record in records:
        prompt = str(record.get("prompt", ""))
        continuation_text = str(record.get("continuation_text", ""))
        generated_text = str(record.get("generated_text", ""))

        overlap_values.append(prompt_to_first_sentence_overlap(prompt, continuation_text))
        sentence_counts.append(len(split_sentences(generated_text)))
        ended_with_eot_flags.append(bool(record.get("ended_with_eot", False)))
        terminal_punctuation_flags.append(bool(continuation_text.rstrip().endswith(TERMINAL_PUNCTUATION)))
        distinct_4_values.append(distinct_4_ratio(continuation_text))
        generated_tokens.append(float(record.get("generated_tokens", 0)))

    sentence_count_distribution = Counter(sentence_counts)
    return {
        "input_jsonl": str(input_path),
        "sample_count": len(records),
        "prompt_to_first_sentence_overlap": mean(overlap_values),
        "sentence_count": mean([float(value) for value in sentence_counts]),
        "sentence_count_distribution": dict(sorted(sentence_count_distribution.items())),
        "ended_with_eot_rate": mean([1.0 if flag else 0.0 for flag in ended_with_eot_flags]),
        "terminal_punctuation_rate": mean([1.0 if flag else 0.0 for flag in terminal_punctuation_flags]),
        "distinct_4_ratio": mean(distinct_4_values),
        "avg_generated_tokens": mean(generated_tokens),
    }


def write_summary(path: Path, fmt: str, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return

    flat_summary = dict(summary)
    flat_summary["sentence_count_distribution"] = json.dumps(summary["sentence_count_distribution"], ensure_ascii=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing sample file: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = ".summary.json" if args.format == "json" else ".summary.csv"
        output_path = input_path.with_name(input_path.stem + suffix)

    records = load_records(input_path)
    summary = summarize_records(records, input_path)
    write_summary(output_path, args.format, summary)
    print(f"Wrote sample summary to {output_path}")


if __name__ == "__main__":
    main()
