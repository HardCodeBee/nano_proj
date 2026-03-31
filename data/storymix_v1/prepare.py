"""
Prepare storymix_v1 for the new from-scratch Task 2 mainline.

Outputs:
- train.bin / val.bin
- train_story_starts.npy / train_story_lengths.npy / train_first_sentence_lengths.npy
- val_story_starts.npy / val_story_lengths.npy / val_first_sentence_lengths.npy
- val_full.txt / locked_test.txt
- dataset_stats.json / filter_report.json

Train is a shuffled mix of:
1. ROCStories train (full held-out-split-compliant train set)
2. A filtered + ranked TinyStories subset

Validation stays ROC-only so checkpoint selection remains aligned to story generation
on the target domain.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


ROC_DATASET_ID = "mintujupally/ROCStories"
TINY_DATASET_ID = "roneneldan/TinyStories"
ROC_TEXT_KEY = "text"
TINY_TEXT_KEY = "text"
ROC_VAL_FRACTION = 0.05
DEFAULT_SEED = 2027
DEFAULT_TINY_TOP_K = 120000
ENC = tiktoken.get_encoding("gpt2")
EOT_TOKEN_ID = int(ENC.eot_token)
TERMINAL_PUNCTUATION = (".", "!", "?")
QUOTE_CHARS = ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019")
QUOTE_NORMALIZATION_MAP = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "`": "'",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LIST_LINE_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
QA_RE = re.compile(r"(?i)(?:^|\n)\s*(?:q|a)\s*[:\-]")
SPEAKER_RE = re.compile(r"(?m)^\s*[A-Z][A-Za-z]{0,20}\s*:\s+")
BRACKETED_STAGE_RE = re.compile(r"\[[^\]]{1,40}\]")

SCENE_KEYWORDS = {
    "home": {"home", "house", "kitchen", "bedroom", "family", "mom", "dad", "parents", "dinner"},
    "school": {"school", "teacher", "class", "classroom", "homework", "student", "students", "lunch"},
    "work": {"work", "office", "boss", "coworker", "meeting", "job", "shift", "store"},
    "social": {"friend", "friends", "party", "neighbor", "together", "visit", "visited", "talked"},
    "everyday": {"park", "bus", "walk", "walked", "bike", "bought", "shopping", "rain", "weekend"},
}
FANTASY_KEYWORDS = {
    "alien",
    "aliens",
    "castle",
    "dragon",
    "dragons",
    "enchanted",
    "fairy",
    "ghost",
    "king",
    "kingdom",
    "magic",
    "magical",
    "monster",
    "monsters",
    "pirate",
    "pirates",
    "princess",
    "robot",
    "robots",
    "spaceship",
    "superhero",
    "treasure",
    "unicorn",
    "wizard",
    "wizards",
}
COMMON_SENTENCE_INITIALS = {
    "After",
    "Before",
    "Later",
    "Soon",
    "Then",
    "When",
    "While",
    "The",
    "A",
    "An",
    "One",
    "Two",
    "That",
    "This",
    "It",
    "He",
    "She",
    "They",
    "We",
    "I",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare storymix_v1 with ROC + filtered TinyStories.")
    parser.add_argument("--roc-dataset-id", default=ROC_DATASET_ID)
    parser.add_argument("--tiny-dataset-id", default=TINY_DATASET_ID)
    parser.add_argument("--roc-text-key", default=ROC_TEXT_KEY)
    parser.add_argument("--tiny-text-key", default=TINY_TEXT_KEY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--roc-val-fraction", type=float, default=ROC_VAL_FRACTION)
    parser.add_argument("--tiny-train-split", default="train")
    parser.add_argument("--max-tiny-train-examples", type=int, default=DEFAULT_TINY_TOP_K)
    parser.add_argument(
        "--max-tiny-source-examples",
        type=int,
        default=None,
        help="Optional dev cap before filtering; default scans the full TinyStories train split.",
    )
    parser.add_argument("--min-sentences", type=int, default=4)
    parser.add_argument("--max-sentences", type=int, default=6)
    parser.add_argument("--min-bpe-len", type=int, default=32)
    parser.add_argument("--max-bpe-len", type=int, default=90)
    parser.add_argument("--max-quote-count", type=int, default=4)
    parser.add_argument("--max-dialogue-sentences", type=int, default=1)
    parser.add_argument("--max-repeat-4gram-ratio", type=float, default=0.18)
    parser.add_argument("--near-dedup-threshold", type=float, default=0.92)
    parser.add_argument("--length-distance-scale", type=float, default=8.0)
    parser.add_argument("--target-sentence-count", type=int, default=5)
    parser.add_argument("--target-sentence-bonus", type=float, default=1.25)
    parser.add_argument("--neighbor-sentence-bonus", type=float, default=0.25)
    parser.add_argument("--scene-bonus", type=float, default=0.45)
    parser.add_argument("--fantasy-penalty", type=float, default=0.55)
    parser.add_argument("--titlecase-penalty", type=float, default=0.35)
    parser.add_argument("--exclamation-penalty", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    return " ".join(text.replace("\n", " ").split()).strip()


def normalize_for_dedup(text: str) -> str:
    text = normalize_whitespace(text)
    for source, target in QUOTE_NORMALIZATION_MAP.items():
        text = text.replace(source, target)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def extract_first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else normalize_whitespace(text)


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def summarize_lengths(lengths: np.ndarray | list[int]) -> dict:
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "p90": int(np.percentile(arr, 90)) if arr.size else 0,
        "p95": int(np.percentile(arr, 95)) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
    }


def write_eval_text(path: Path, stories: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n\n".join(story.strip() for story in stories if story.strip()))
        handle.write("\n")


def save_story_metadata(
    out_dir: Path,
    split: str,
    starts: np.ndarray,
    lengths: np.ndarray,
    first_sentence_lengths: np.ndarray,
) -> None:
    np.save(out_dir / f"{split}_story_starts.npy", starts.astype(np.int64, copy=False))
    np.save(out_dir / f"{split}_story_lengths.npy", lengths.astype(np.int64, copy=False))
    np.save(
        out_dir / f"{split}_first_sentence_lengths.npy",
        first_sentence_lengths.astype(np.int64, copy=False),
    )


def compute_repeat_4gram_ratio(tokens: list[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    ngrams = Counter(tuple(tokens[idx : idx + 4]) for idx in range(len(tokens) - 3))
    repeated = sum(count - 1 for count in ngrams.values() if count > 1)
    return repeated / max(1, len(tokens) - 3)


def looks_heavily_formatted(text: str) -> bool:
    if LIST_LINE_RE.search(text):
        return True
    if QA_RE.search(text):
        return True
    if SPEAKER_RE.search(text):
        return True
    if BRACKETED_STAGE_RE.search(text):
        return True
    if text.count("|") >= 2 or "\t" in text:
        return True
    return False


def count_dialogue_sentences(sentences: list[str]) -> int:
    count = 0
    for sentence in sentences:
        stripped = sentence.lstrip()
        if stripped.startswith(tuple(QUOTE_CHARS)) or re.match(r"^[A-Z][A-Za-z]{0,20}\s*:", stripped):
            count += 1
    return count


def count_noninitial_titlecase_tokens(text: str) -> int:
    count = 0
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        token = match.group(0)
        if token in COMMON_SENTENCE_INITIALS:
            continue
        prefix = text[: match.start()].rstrip()
        if not prefix or prefix.endswith(TERMINAL_PUNCTUATION):
            continue
        count += 1
    return count


def soft_score_story(
    text: str,
    bpe_len: int,
    sentence_count: int,
    roc_length_median: float,
    args: argparse.Namespace,
) -> float:
    lowered_tokens = set(word_tokens(text))
    score = 0.0

    score -= abs(bpe_len - roc_length_median) / args.length_distance_scale
    if sentence_count == args.target_sentence_count:
        score += args.target_sentence_bonus
    elif abs(sentence_count - args.target_sentence_count) == 1:
        score += args.neighbor_sentence_bonus

    scene_hits = sum(1 for words in SCENE_KEYWORDS.values() if lowered_tokens & words)
    score += args.scene_bonus * scene_hits

    fantasy_hits = len(lowered_tokens & FANTASY_KEYWORDS)
    score -= args.fantasy_penalty * min(fantasy_hits, 4)

    titlecase_count = count_noninitial_titlecase_tokens(text)
    if titlecase_count > 1:
        score -= args.titlecase_penalty * (titlecase_count - 1)

    exclamation_count = text.count("!")
    if exclamation_count > 1:
        score -= args.exclamation_penalty * (exclamation_count - 1)

    return score


def build_simhash(tokens: list[str]) -> int:
    shingles = [" ".join(tokens[idx : idx + 4]) for idx in range(len(tokens) - 3)]
    if not shingles:
        shingles = tokens or [""]

    accumulator = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit_idx in range(64):
            accumulator[bit_idx] += 1 if (digest >> bit_idx) & 1 else -1

    value = 0
    for bit_idx, bit_value in enumerate(accumulator):
        if bit_value >= 0:
            value |= 1 << bit_idx
    return value


def candidate_sort_key(candidate: dict) -> tuple[float, float, int]:
    return (
        candidate["soft_score"],
        -abs(candidate["bpe_len"] - candidate["roc_length_median"]),
        -candidate["original_index"],
    )


def build_near_dedup_keys(candidate: dict) -> list[tuple]:
    tokens = candidate["tokens"]
    length_bucket = candidate["bpe_len"] // 8
    first_key = " ".join(tokens[:8])
    last_key = " ".join(tokens[-8:])
    simhash = candidate["simhash"]
    return [
        ("first", candidate["sentence_count"], length_bucket, first_key),
        ("last", candidate["sentence_count"], length_bucket, last_key),
        ("sim_hi", candidate["sentence_count"], length_bucket, simhash >> 48),
        ("sim_mid", candidate["sentence_count"], length_bucket, (simhash >> 24) & 0xFFFF),
        ("sim_lo", candidate["sentence_count"], length_bucket, simhash & 0xFFFF),
    ]


def is_near_duplicate(
    candidate: dict,
    accepted: list[dict],
    buckets: dict[tuple, list[int]],
    threshold: float,
) -> bool:
    candidate_keys = build_near_dedup_keys(candidate)
    seen_indices: set[int] = set()
    for key in candidate_keys:
        for accepted_idx in buckets.get(key, []):
            if accepted_idx in seen_indices:
                continue
            seen_indices.add(accepted_idx)
            other = accepted[accepted_idx]
            if abs(candidate["bpe_len"] - other["bpe_len"]) > 8:
                continue
            ratio = difflib.SequenceMatcher(None, candidate["canonical_text"], other["canonical_text"]).ratio()
            if ratio >= threshold:
                return True
            if candidate["tokens"][:10] == other["tokens"][:10] and candidate["tokens"][-10:] == other["tokens"][-10:]:
                return True
    return False


def tokenize_story(text: str) -> tuple[list[int], int, int]:
    ids = ENC.encode_ordinary(text)
    first_sentence = extract_first_sentence(text)
    first_sentence_len = len(ENC.encode_ordinary(first_sentence)) if first_sentence else 0
    ids.append(EOT_TOKEN_ID)
    return ids, len(ids), first_sentence_len


def build_split_artifacts(
    out_dir: Path,
    split: str,
    stories: list[str],
    sources: list[str] | None = None,
    write_bin: bool = True,
) -> dict:
    lengths = np.empty(len(stories), dtype=np.int64)
    first_sentence_lengths = np.empty(len(stories), dtype=np.int64)
    source_story_counts = Counter()
    source_token_totals = Counter()

    for idx, story in enumerate(stories):
        _, story_len, first_sentence_len = tokenize_story(story)
        lengths[idx] = story_len
        first_sentence_lengths[idx] = first_sentence_len
        if sources is not None:
            source_story_counts[sources[idx]] += 1
            source_token_totals[sources[idx]] += story_len

    stats = summarize_lengths(lengths)
    stats["stories"] = len(stories)
    stats["tokens_total"] = int(lengths.sum(dtype=np.int64))
    if sources is not None:
        stats["source_story_counts"] = dict(sorted(source_story_counts.items()))
        stats["source_token_totals"] = dict(sorted(source_token_totals.items()))

    if write_bin:
        starts = np.cumsum(np.concatenate(([0], lengths[:-1])), dtype=np.int64) if len(lengths) else np.array([], dtype=np.int64)
        arr = np.memmap(
            out_dir / f"{split}.bin",
            dtype=np.uint16,
            mode="w+",
            shape=(stats["tokens_total"],),
        )
        cursor = 0
        for story in stories:
            ids, _, _ = tokenize_story(story)
            story_arr = np.asarray(ids, dtype=np.uint16)
            arr[cursor : cursor + len(story_arr)] = story_arr
            cursor += len(story_arr)
        arr.flush()
        save_story_metadata(out_dir, split, starts, lengths, first_sentence_lengths)

    return stats


def build_roc_split_map(dataset, text_key: str, val_fraction: float, seed: int) -> dict[str, list[str]]:
    train_split = dataset["train"].train_test_split(
        test_size=val_fraction,
        seed=seed,
        shuffle=True,
    )
    return {
        "train": [normalize_whitespace(text) for text in train_split["train"][text_key]],
        "val": [normalize_whitespace(text) for text in train_split["test"][text_key]],
        "locked_test": [normalize_whitespace(text) for text in dataset["test"][text_key]],
    }


def build_roc_reference_sets(roc_splits: dict[str, list[str]]) -> tuple[set[str], dict[str, set[str]]]:
    full_text_set: set[str] = set()
    first_sentence_sets: dict[str, set[str]] = {}
    for split, stories in roc_splits.items():
        normalized_stories = {normalize_for_dedup(story) for story in stories}
        full_text_set.update(normalized_stories)
        first_sentence_sets[split] = {normalize_for_dedup(extract_first_sentence(story)) for story in stories}
    return full_text_set, first_sentence_sets


def inspect_tiny_story(text: str, args: argparse.Namespace, roc_length_median: float) -> tuple[dict | None, str | None]:
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return None, "empty"

    sentences = split_sentences(normalized_text)
    sentence_count = len(sentences)
    if sentence_count < args.min_sentences or sentence_count > args.max_sentences:
        return None, "sentence_count"

    if not sentences[-1].endswith(TERMINAL_PUNCTUATION):
        return None, "terminal_punctuation"

    if looks_heavily_formatted(text):
        return None, "formatted_or_script"

    quote_count = sum(normalized_text.count(ch) for ch in QUOTE_CHARS)
    if quote_count > args.max_quote_count:
        return None, "quote_heavy"

    dialogue_sentences = count_dialogue_sentences(sentences)
    if dialogue_sentences > args.max_dialogue_sentences:
        return None, "dialogue_heavy"

    tokens = word_tokens(normalized_text)
    repeat_4gram_ratio = compute_repeat_4gram_ratio(tokens)
    if repeat_4gram_ratio > args.max_repeat_4gram_ratio:
        return None, "repeat_4gram"

    sentence_keys = [normalize_for_dedup(sentence) for sentence in sentences]
    if len(sentence_keys) != len(set(sentence_keys)):
        return None, "duplicate_sentence"

    bpe_len = len(ENC.encode_ordinary(normalized_text))
    if bpe_len < args.min_bpe_len or bpe_len > args.max_bpe_len:
        return None, "bpe_length"

    first_sentence = extract_first_sentence(normalized_text)
    candidate = {
        "text": normalized_text,
        "normalized_text": normalize_for_dedup(normalized_text),
        "first_sentence": first_sentence,
        "normalized_first_sentence": normalize_for_dedup(first_sentence),
        "tokens": tokens,
        "canonical_text": " ".join(tokens),
        "sentence_count": sentence_count,
        "bpe_len": bpe_len,
        "repeat_4gram_ratio": repeat_4gram_ratio,
        "soft_score": soft_score_story(normalized_text, bpe_len, sentence_count, roc_length_median, args),
        "roc_length_median": roc_length_median,
        "simhash": build_simhash(tokens),
    }
    return candidate, None


def collect_tiny_candidates(
    tiny_dataset,
    args: argparse.Namespace,
    roc_text_set: set[str],
    roc_first_sentence_sets: dict[str, set[str]],
    roc_length_median: float,
) -> tuple[list[dict], dict]:
    hard_failures = Counter()
    leakage_failures = Counter()
    exact_dedup_collisions = 0
    unique_exact: dict[str, dict] = {}

    total_source_examples = len(tiny_dataset)
    for idx, row in enumerate(tiny_dataset):
        if args.progress_every > 0 and idx > 0 and idx % args.progress_every == 0:
            print(
                f"[storymix_v1] scanned {idx:,}/{total_source_examples:,} TinyStories rows; "
                f"current unique hard-pass pool={len(unique_exact):,}"
            )

        candidate, hard_fail_reason = inspect_tiny_story(row[args.tiny_text_key], args, roc_length_median)
        if hard_fail_reason is not None:
            hard_failures[hard_fail_reason] += 1
            continue

        normalized_text = candidate["normalized_text"]
        normalized_first_sentence = candidate["normalized_first_sentence"]
        if normalized_text in roc_text_set:
            leakage_failures["roc_exact_text_match"] += 1
            continue
        if normalized_first_sentence in roc_first_sentence_sets["val"]:
            leakage_failures["roc_val_first_sentence_match"] += 1
            continue
        if normalized_first_sentence in roc_first_sentence_sets["locked_test"]:
            leakage_failures["roc_locked_test_first_sentence_match"] += 1
            continue
        if normalized_first_sentence in roc_first_sentence_sets["train"]:
            leakage_failures["roc_train_first_sentence_match"] += 1
            continue

        candidate["original_index"] = idx
        previous = unique_exact.get(normalized_text)
        if previous is None:
            unique_exact[normalized_text] = candidate
            continue

        exact_dedup_collisions += 1
        if candidate_sort_key(candidate) > candidate_sort_key(previous):
            unique_exact[normalized_text] = candidate

    ranked_candidates = sorted(unique_exact.values(), key=candidate_sort_key, reverse=True)
    accepted: list[dict] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    near_dedup_removed = 0

    for candidate in ranked_candidates:
        if is_near_duplicate(candidate, accepted, buckets, args.near_dedup_threshold):
            near_dedup_removed += 1
            continue
        accepted.append(candidate)
        candidate_idx = len(accepted) - 1
        for key in build_near_dedup_keys(candidate):
            buckets[key].append(candidate_idx)
        if len(accepted) >= args.max_tiny_train_examples:
            break

    kept_lengths = [candidate["bpe_len"] for candidate in accepted]
    kept_sentence_counts = Counter(candidate["sentence_count"] for candidate in accepted)
    kept_soft_scores = np.asarray([candidate["soft_score"] for candidate in accepted], dtype=np.float64)
    filter_report = {
        "tiny_source_examples_scanned": total_source_examples,
        "hard_filter_failures": dict(sorted(hard_failures.items())),
        "leakage_guard_failures": dict(sorted(leakage_failures.items())),
        "exact_dedup_collisions": exact_dedup_collisions,
        "hard_pass_unique_exact_pool": len(unique_exact),
        "near_dedup_removed": near_dedup_removed,
        "selected_tiny_examples": len(accepted),
        "selection_target": args.max_tiny_train_examples,
        "soft_rank_tail_dropped": max(0, len(ranked_candidates) - near_dedup_removed - len(accepted)),
        "selected_bpe_length_summary": summarize_lengths(kept_lengths),
        "selected_sentence_count_distribution": dict(sorted(kept_sentence_counts.items())),
        "selected_soft_score_summary": {
            "min": float(kept_soft_scores.min()) if kept_soft_scores.size else 0.0,
            "mean": float(kept_soft_scores.mean()) if kept_soft_scores.size else 0.0,
            "median": float(np.median(kept_soft_scores)) if kept_soft_scores.size else 0.0,
            "max": float(kept_soft_scores.max()) if kept_soft_scores.size else 0.0,
        },
    }
    return accepted, filter_report


def main() -> None:
    args = parse_args()
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[storymix_v1] loading ROCStories splits...")
    roc_dataset = load_dataset(args.roc_dataset_id)
    roc_splits = build_roc_split_map(
        roc_dataset,
        text_key=args.roc_text_key,
        val_fraction=args.roc_val_fraction,
        seed=args.seed,
    )
    roc_text_set, roc_first_sentence_sets = build_roc_reference_sets(roc_splits)
    roc_train_lengths = [len(ENC.encode_ordinary(story)) + 1 for story in roc_splits["train"]]
    roc_length_median = float(np.median(np.asarray(roc_train_lengths, dtype=np.int64)))

    print("[storymix_v1] loading TinyStories train split...")
    tiny_train = load_dataset(args.tiny_dataset_id, split=args.tiny_train_split)
    if args.max_tiny_source_examples is not None and args.max_tiny_source_examples > 0:
        tiny_train = tiny_train.select(range(min(len(tiny_train), args.max_tiny_source_examples)))

    selected_tiny, filter_report = collect_tiny_candidates(
        tiny_dataset=tiny_train,
        args=args,
        roc_text_set=roc_text_set,
        roc_first_sentence_sets=roc_first_sentence_sets,
        roc_length_median=roc_length_median,
    )

    roc_train_records = [(story, "rocstories") for story in roc_splits["train"]]
    tiny_records = [(candidate["text"], "tinystories") for candidate in selected_tiny]
    mixed_records = roc_train_records + tiny_records

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(mixed_records))
    train_stories = [mixed_records[idx][0] for idx in order.tolist()]
    train_sources = [mixed_records[idx][1] for idx in order.tolist()]

    val_stories = list(roc_splits["val"])
    val_sources = ["rocstories"] * len(val_stories)
    locked_test_stories = list(roc_splits["locked_test"])

    print("[storymix_v1] writing train/val token streams and metadata...")
    train_stats = build_split_artifacts(out_dir, "train", train_stories, train_sources, write_bin=True)
    val_stats = build_split_artifacts(out_dir, "val", val_stories, val_sources, write_bin=True)
    locked_test_lengths = [len(ENC.encode_ordinary(story)) + 1 for story in locked_test_stories]
    locked_test_stats = summarize_lengths(locked_test_lengths)
    locked_test_stats["stories"] = len(locked_test_stories)
    locked_test_stats["tokens_total"] = int(np.sum(np.asarray(locked_test_lengths, dtype=np.int64), dtype=np.int64))
    locked_test_stats["source_story_counts"] = {"rocstories": len(locked_test_stories)}
    locked_test_stats["source_token_totals"] = {"rocstories": locked_test_stats["tokens_total"]}

    write_eval_text(out_dir / "val_full.txt", val_stories)
    write_eval_text(out_dir / "locked_test.txt", locked_test_stories)

    dataset_stats = {
        "dataset_id": "storymix_v1",
        "tokenizer": "gpt2",
        "separator_token_id": EOT_TOKEN_ID,
        "split_policy": {
            "train_sources": {
                "rocstories_train": "official_train minus held-out val",
                "tinystories_train": "filtered/ranked TinyStories train subset",
            },
            "val_source": "ROCStories held-out val only",
            "locked_test_source": "ROCStories official public test only",
            "roc_val_fraction": args.roc_val_fraction,
            "seed": args.seed,
            "train_shuffle_seed": args.seed,
        },
        "tiny_filter_defaults": {
            "sentence_range": [args.min_sentences, args.max_sentences],
            "bpe_length_range": [args.min_bpe_len, args.max_bpe_len],
            "max_quote_count": args.max_quote_count,
            "max_dialogue_sentences": args.max_dialogue_sentences,
            "max_repeat_4gram_ratio": args.max_repeat_4gram_ratio,
            "near_dedup_threshold": args.near_dedup_threshold,
            "top_tiny_examples": args.max_tiny_train_examples,
        },
        "tiny_sort_defaults": {
            "length_distance_scale": args.length_distance_scale,
            "target_sentence_count": args.target_sentence_count,
            "target_sentence_bonus": args.target_sentence_bonus,
            "neighbor_sentence_bonus": args.neighbor_sentence_bonus,
            "scene_bonus": args.scene_bonus,
            "fantasy_penalty": args.fantasy_penalty,
            "titlecase_penalty": args.titlecase_penalty,
            "exclamation_penalty": args.exclamation_penalty,
        },
        "mixture": {
            "rocstories_train_stories": len(roc_splits["train"]),
            "tinystories_selected_stories": len(selected_tiny),
            "total_train_stories": len(train_stories),
        },
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "locked_test": locked_test_stats,
        },
    }

    with (out_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_stats, handle, indent=2)
    with (out_dir / "filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(filter_report, handle, indent=2)

    print(
        "[storymix_v1] done: "
        f"train stories={len(train_stories):,} "
        f"(roc={len(roc_splits['train']):,}, tiny={len(selected_tiny):,}), "
        f"val stories={len(val_stories):,}"
    )


if __name__ == "__main__":
    main()
