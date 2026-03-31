"""
Prepare a narrow ROC-style TinyStories subset for short curriculum pretraining.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
HELPER_PATH = ROOT.parent / "storymix_v1" / "prepare.py"
HELPER_SPEC = importlib.util.spec_from_file_location("storymix_v1_prepare_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load shared helpers from {HELPER_PATH}")
helpers = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helpers)

ROC_DATASET_ID = helpers.ROC_DATASET_ID
TINY_DATASET_ID = helpers.TINY_DATASET_ID
ROC_TEXT_KEY = helpers.ROC_TEXT_KEY
TINY_TEXT_KEY = helpers.TINY_TEXT_KEY
ENC = helpers.ENC
EOT_TOKEN_ID = helpers.EOT_TOKEN_ID
TERMINAL_PUNCTUATION = helpers.TERMINAL_PUNCTUATION
DOUBLE_QUOTE_CHARS = ('"', "\u201c", "\u201d")
SPEAKER_RE = re.compile(r"^[A-Z][A-Za-z]{0,20}\s*:")
FORMAT_RE = re.compile(r"(?im)(^\s*(?:[-*]|\d+[.)])\s+|(?:^|\n)\s*(?:q|a)\s*[:\-]|^\s*[A-Z][A-Za-z]{0,20}\s*:\s+)")
METADATA_RE = re.compile(
    r"(?i)(<[^>]+>|https?://|www\.|(?:title|summary|genre|tags?|keywords?|chapter|episode|scene|prompt|metadata|author)\s*:|table of contents|copyright|all rights reserved|```|###)"
)
SCENE_KEYWORDS = [
    set("home house kitchen bedroom family mom dad parents dinner".split()),
    set("school teacher class classroom homework student students lunch".split()),
    set("work office boss coworker coworkers meeting job shift store".split()),
    set("friend friends party neighbor neighbors together visit visited talked".split()),
    set("park bus walk walked bike bought shopping rain weekend".split()),
]
FANTASY_KEYWORDS = set(
    "alien aliens castle dragon dragons enchanted fairy ghost king kingdom magic magical monster monsters pirate pirates princess prince robot robots spaceship superhero treasure unicorn wizard wizards".split()
)
WORLD_KEYWORDS = set("empire emperor galaxy kingdom portal prophecy quest spaceship spell spells throne wizard wizards".split())
ADVENTURE_KEYWORDS = set("adventure adventures battle captain cave explore explored forest island journey mountain quest rescued sail sailed sword treasure".split())
WEIRD_KEYWORDS = set("alien aliens galaxy haunted monster monsters planet portal robot robots spaceship superhero".split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare filtered ROC-style TinyStories for Stage A.")
    parser.add_argument("--roc-dataset-id", default=ROC_DATASET_ID)
    parser.add_argument("--tiny-dataset-id", default=TINY_DATASET_ID)
    parser.add_argument("--roc-text-key", default=ROC_TEXT_KEY)
    parser.add_argument("--tiny-text-key", default=TINY_TEXT_KEY)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--roc-val-fraction", type=float, default=0.05)
    parser.add_argument("--tiny-train-split", default="train")
    parser.add_argument("--tiny-val-split", default="validation")
    parser.add_argument("--max-train-examples", type=int, default=30000)
    parser.add_argument("--max-val-examples", type=int, default=3000)
    parser.add_argument("--max-train-source-examples", type=int, default=None)
    parser.add_argument("--max-val-source-examples", type=int, default=None)
    parser.add_argument("--min-sentences", type=int, default=4)
    parser.add_argument("--max-sentences", type=int, default=6)
    parser.add_argument("--min-bpe-len", type=int, default=32)
    parser.add_argument("--max-bpe-len", type=int, default=80)
    parser.add_argument("--max-double-quote-count", type=int, default=2)
    parser.add_argument("--max-dialogue-sentences", type=int, default=1)
    parser.add_argument("--max-repeat-4gram-ratio", type=float, default=0.12)
    parser.add_argument("--near-dedup-threshold", type=float, default=0.92)
    parser.add_argument("--max-noninitial-titlecase-tokens", type=int, default=3)
    parser.add_argument("--max-fantasy-keyword-hits", type=int, default=2)
    parser.add_argument("--max-worldbuilding-keyword-hits", type=int, default=2)
    parser.add_argument("--length-distance-scale", type=float, default=8.0)
    parser.add_argument("--target-sentence-count", type=int, default=5)
    parser.add_argument("--target-sentence-bonus", type=float, default=1.25)
    parser.add_argument("--neighbor-sentence-bonus", type=float, default=0.25)
    parser.add_argument("--scene-bonus", type=float, default=0.45)
    parser.add_argument("--fantasy-penalty", type=float, default=0.60)
    parser.add_argument("--adventure-penalty", type=float, default=0.20)
    parser.add_argument("--weird-setting-penalty", type=float, default=0.30)
    parser.add_argument("--titlecase-penalty", type=float, default=0.30)
    parser.add_argument("--quote-penalty", type=float, default=0.20)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def maybe_limit(dset, limit):
    if limit is None or limit <= 0 or len(dset) <= limit:
        return dset
    return dset.select(range(limit))


def count_dialogue_sentences(sentences):
    total = 0
    for sentence in sentences:
        stripped = sentence.lstrip()
        if stripped.startswith(DOUBLE_QUOTE_CHARS) or SPEAKER_RE.match(stripped):
            total += 1
    return total


def soft_score(text, bpe_len, sentence_count, roc_len, args):
    lowered = set(helpers.word_tokens(text))
    score = -abs(bpe_len - roc_len) / args.length_distance_scale
    if sentence_count == args.target_sentence_count:
        score += args.target_sentence_bonus
    elif abs(sentence_count - args.target_sentence_count) == 1:
        score += args.neighbor_sentence_bonus
    score += args.scene_bonus * sum(1 for words in SCENE_KEYWORDS if lowered & words)
    score -= args.fantasy_penalty * min(len(lowered & FANTASY_KEYWORDS), 4)
    score -= args.adventure_penalty * min(len(lowered & ADVENTURE_KEYWORDS), 4)
    score -= args.weird_setting_penalty * min(len(lowered & WEIRD_KEYWORDS), 4)
    titlecase_count = helpers.count_noninitial_titlecase_tokens(text)
    if titlecase_count > 1:
        score -= args.titlecase_penalty * (titlecase_count - 1)
    quote_count = sum(text.count(ch) for ch in DOUBLE_QUOTE_CHARS)
    if quote_count > 0:
        score -= args.quote_penalty * min(quote_count, 3)
    return score


def inspect_story(text, args, roc_len):
    text = helpers.normalize_whitespace(text)
    if not text:
        return None, "empty"
    sentences = helpers.split_sentences(text)
    if not (args.min_sentences <= len(sentences) <= args.max_sentences):
        return None, "sentence_count"
    if not sentences[-1].endswith(TERMINAL_PUNCTUATION):
        return None, "terminal_punctuation"
    if FORMAT_RE.search(text) or helpers.BRACKETED_STAGE_RE.search(text):
        return None, "formatted_or_script"
    if METADATA_RE.search(text):
        return None, "metadata_html_or_nonnarrative"
    quote_count = sum(text.count(ch) for ch in DOUBLE_QUOTE_CHARS)
    if quote_count > args.max_double_quote_count:
        return None, "quote_heavy"
    if count_dialogue_sentences(sentences) > args.max_dialogue_sentences:
        return None, "dialogue_heavy"
    tokens = helpers.word_tokens(text)
    if helpers.compute_repeat_4gram_ratio(tokens) > args.max_repeat_4gram_ratio:
        return None, "repeat_4gram"
    sentence_keys = [helpers.normalize_for_dedup(sentence) for sentence in sentences]
    if len(sentence_keys) != len(set(sentence_keys)):
        return None, "duplicate_sentence"
    bpe_len = len(ENC.encode_ordinary(text))
    if not (args.min_bpe_len <= bpe_len <= args.max_bpe_len):
        return None, "bpe_length"
    titlecase_count = helpers.count_noninitial_titlecase_tokens(text)
    if titlecase_count > args.max_noninitial_titlecase_tokens:
        return None, "named_entity_heavy"
    lowered = set(tokens)
    if len(lowered & FANTASY_KEYWORDS) >= args.max_fantasy_keyword_hits:
        return None, "fantasy_heavy"
    if len(lowered & WORLD_KEYWORDS) >= args.max_worldbuilding_keyword_hits:
        return None, "worldbuilding_heavy"
    return {
        "text": text,
        "normalized_text": helpers.normalize_for_dedup(text),
        "normalized_first_sentence": helpers.normalize_for_dedup(helpers.extract_first_sentence(text)),
        "tokens": tokens,
        "canonical_text": " ".join(tokens),
        "sentence_count": len(sentences),
        "bpe_len": bpe_len,
        "soft_score": soft_score(text, bpe_len, len(sentences), roc_len, args),
        "roc_length_median": roc_len,
        "target_sentence_count": args.target_sentence_count,
        "simhash": helpers.build_simhash(tokens),
    }, None


def collect_candidates(name, tiny_dataset, args, max_examples, text_key, roc_text_set, roc_first_sentence_sets, roc_len):
    hard_failures, leakage_failures, unique_exact = Counter(), Counter(), {}
    exact_dedup_collisions = 0
    for idx, row in enumerate(tiny_dataset):
        if args.progress_every > 0 and idx > 0 and idx % args.progress_every == 0:
            print(f"[tinystories_rocstyle_v2:{name}] scanned {idx:,}/{len(tiny_dataset):,} rows; hard-pass pool={len(unique_exact):,}")
        candidate, reason = inspect_story(row[text_key], args, roc_len)
        if reason is not None:
            hard_failures[reason] += 1
            continue
        first_sentence = candidate["normalized_first_sentence"]
        if candidate["normalized_text"] in roc_text_set:
            leakage_failures["roc_exact_text_match"] += 1
            continue
        if first_sentence in roc_first_sentence_sets["val"]:
            leakage_failures["roc_val_first_sentence_match"] += 1
            continue
        if first_sentence in roc_first_sentence_sets["locked_test"]:
            leakage_failures["roc_locked_test_first_sentence_match"] += 1
            continue
        if first_sentence in roc_first_sentence_sets["train"]:
            leakage_failures["roc_train_first_sentence_match"] += 1
            continue
        candidate["original_index"] = idx
        previous = unique_exact.get(candidate["normalized_text"])
        if previous is not None:
            exact_dedup_collisions += 1
        if previous is None or (candidate["soft_score"], -candidate["original_index"]) > (previous["soft_score"], -previous["original_index"]):
            unique_exact[candidate["normalized_text"]] = candidate
    ranked = sorted(unique_exact.values(), key=lambda c: (c["soft_score"], -abs(c["bpe_len"] - c["roc_length_median"]), -abs(c["sentence_count"] - c["target_sentence_count"]), -c["original_index"]), reverse=True)
    accepted, buckets, near_dedup_removed = [], defaultdict(list), 0
    for candidate in ranked:
        if helpers.is_near_duplicate(candidate, accepted, buckets, args.near_dedup_threshold):
            near_dedup_removed += 1
            continue
        accepted.append(candidate)
        idx = len(accepted) - 1
        for key in helpers.build_near_dedup_keys(candidate):
            buckets[key].append(idx)
        if max_examples and len(accepted) >= max_examples:
            break
    soft_scores = np.asarray([c["soft_score"] for c in accepted], dtype=np.float64)
    return accepted, {
        "source_examples_scanned": len(tiny_dataset),
        "hard_filter_failures": dict(sorted(hard_failures.items())),
        "leakage_guard_failures": dict(sorted(leakage_failures.items())),
        "exact_dedup_collisions": exact_dedup_collisions,
        "hard_pass_unique_exact_pool": len(unique_exact),
        "near_dedup_removed": near_dedup_removed,
        "selected_examples": len(accepted),
        "selection_target": max_examples,
        "soft_rank_tail_dropped": max(0, len(ranked) - near_dedup_removed - len(accepted)),
        "selected_bpe_length_summary": helpers.summarize_lengths([c["bpe_len"] for c in accepted]),
        "selected_sentence_count_distribution": dict(sorted(Counter(c["sentence_count"] for c in accepted).items())),
        "selected_soft_score_summary": {
            "min": float(soft_scores.min()) if soft_scores.size else 0.0,
            "mean": float(soft_scores.mean()) if soft_scores.size else 0.0,
            "median": float(np.median(soft_scores)) if soft_scores.size else 0.0,
            "max": float(soft_scores.max()) if soft_scores.size else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    roc_dataset = load_dataset(args.roc_dataset_id)
    roc_splits = helpers.build_roc_split_map(roc_dataset, args.roc_text_key, args.roc_val_fraction, args.seed)
    roc_text_set, roc_first_sentence_sets = helpers.build_roc_reference_sets(roc_splits)
    roc_lengths = [len(ENC.encode_ordinary(story)) + 1 for story in roc_splits["train"]]
    roc_len = float(np.median(np.asarray(roc_lengths, dtype=np.int64)))
    tiny_dataset = load_dataset(args.tiny_dataset_id)
    train_split = maybe_limit(tiny_dataset[args.tiny_train_split], args.max_train_source_examples)
    val_split = maybe_limit(tiny_dataset[args.tiny_val_split], args.max_val_source_examples)
    selected_train, train_report = collect_candidates("train", train_split, args, args.max_train_examples, args.tiny_text_key, roc_text_set, roc_first_sentence_sets, roc_len)
    selected_val, val_report = collect_candidates("val", val_split, args, args.max_val_examples, args.tiny_text_key, roc_text_set, roc_first_sentence_sets, roc_len)
    if not selected_train or not selected_val:
        raise RuntimeError("Filtering kept zero examples for train or val; relax the TinyStories filters.")
    train_stories = [c["text"] for c in selected_train]
    val_stories = [c["text"] for c in selected_val]
    train_stats = helpers.build_split_artifacts(ROOT, "train", train_stories, ["tinystories"] * len(train_stories), write_bin=True)
    val_stats = helpers.build_split_artifacts(ROOT, "val", val_stories, ["tinystories"] * len(val_stories), write_bin=True)
    helpers.write_eval_text(ROOT / "val_full.txt", val_stories)
    dataset_stats = {
        "dataset_id": "tinystories_rocstyle_v2",
        "tokenizer": "gpt2",
        "separator_token_id": EOT_TOKEN_ID,
        "split_policy": {
            "train_source": "TinyStories filtered train split only",
            "val_source": "TinyStories filtered validation split only",
            "roc_reference_usage": "soft ranking and leakage guards only; no ROC text written",
            "seed": args.seed,
            "roc_val_fraction_for_reference": args.roc_val_fraction,
        },
        "roc_reference": {
            "dataset_id": args.roc_dataset_id,
            "train_story_count": len(roc_splits["train"]),
            "val_story_count": len(roc_splits["val"]),
            "locked_test_story_count": len(roc_splits["locked_test"]),
            "train_bpe_length_summary": helpers.summarize_lengths(roc_lengths),
            "train_bpe_length_median": roc_len,
        },
        "tiny_filter_defaults": {
            "sentence_range": [args.min_sentences, args.max_sentences],
            "bpe_length_range": [args.min_bpe_len, args.max_bpe_len],
            "max_double_quote_count": args.max_double_quote_count,
            "max_dialogue_sentences": args.max_dialogue_sentences,
            "max_repeat_4gram_ratio": args.max_repeat_4gram_ratio,
            "max_noninitial_titlecase_tokens": args.max_noninitial_titlecase_tokens,
            "max_fantasy_keyword_hits": args.max_fantasy_keyword_hits,
            "max_worldbuilding_keyword_hits": args.max_worldbuilding_keyword_hits,
            "near_dedup_threshold": args.near_dedup_threshold,
            "top_train_examples": args.max_train_examples,
            "top_val_examples": args.max_val_examples,
        },
        "tiny_sort_defaults": {
            "length_distance_scale": args.length_distance_scale,
            "target_sentence_count": args.target_sentence_count,
            "target_sentence_bonus": args.target_sentence_bonus,
            "neighbor_sentence_bonus": args.neighbor_sentence_bonus,
            "scene_bonus": args.scene_bonus,
            "fantasy_penalty": args.fantasy_penalty,
            "adventure_penalty": args.adventure_penalty,
            "weird_setting_penalty": args.weird_setting_penalty,
            "titlecase_penalty": args.titlecase_penalty,
            "quote_penalty": args.quote_penalty,
        },
        "splits": {"train": train_stats, "val": val_stats},
    }
    filter_report = {"dataset_id": "tinystories_rocstyle_v2", "reference_roc_length_median": roc_len, "train": train_report, "val": val_report}
    (ROOT / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2) + "\n", encoding="utf-8")
    (ROOT / "filter_report.json").write_text(json.dumps(filter_report, indent=2) + "\n", encoding="utf-8")
    print(f"[tinystories_rocstyle_v2] done: train={len(train_stories):,}, val={len(val_stories):,}, roc_ref_median={roc_len:.1f}")


if __name__ == "__main__":
    main()
