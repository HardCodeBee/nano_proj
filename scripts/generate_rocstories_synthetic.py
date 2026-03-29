"""
Generate ROCStories-style synthetic stories from a source narrative dataset using an
OpenAI-compatible API.

Typical usage (bash):
./.venv/bin/python scripts/generate_rocstories_synthetic.py \
  --api-key "$OPENAI_API_KEY" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o-mini" \
  --target-count 3000 \
  --source-limit 12000 \
  --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import tiktoken
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "task2_rocstyle_rewrite_prompt.txt"
DEFAULT_ACCEPTED = PROJECT_ROOT / "data" / "rocstories_synth" / "raw" / "accepted.jsonl"
DEFAULT_REJECTED = PROJECT_ROOT / "data" / "rocstories_synth" / "raw" / "rejected.jsonl"
DEFAULT_DATASET_ID = "roneneldan/TinyStories"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ENC = tiktoken.get_encoding("gpt2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROCStories-style synthetic stories.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--source-dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--source-text-key", default="text")
    parser.add_argument("--source-limit", type=int, default=12000)
    parser.add_argument("--target-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-source-chars", type=int, default=1600)
    parser.add_argument("--accepted-jsonl", default=str(DEFAULT_ACCEPTED))
    parser.add_argument("--rejected-jsonl", default=str(DEFAULT_REJECTED))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.replace("\n", " ").split()).strip()
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def has_repeated_4gram(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < 8:
        return False
    seen = set()
    for idx in range(len(tokens) - 3):
        gram = tuple(tokens[idx : idx + 4])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def validate_story(story: str) -> tuple[bool, str, int, int]:
    normalized = " ".join(story.split()).strip()
    sentences = split_sentences(normalized)
    token_count = len(ENC.encode_ordinary(normalized))
    if len(sentences) != 5:
        return False, f"expected_5_sentences_got_{len(sentences)}", len(sentences), token_count
    if token_count < 35 or token_count > 110:
        return False, f"token_count_out_of_range_{token_count}", len(sentences), token_count
    if '"' in normalized or "“" in normalized or "”" in normalized:
        return False, "quote_heavy_output", len(sentences), token_count
    if has_repeated_4gram(normalized):
        return False, "repeated_4gram", len(sentences), token_count
    return True, "accepted", len(sentences), token_count


def generate_one_story(
    source_story: str,
    system_prompt: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> str:
    url = normalize_base_url(base_url)
    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source_story},
        ],
    }
    request_body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, data=request_body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            response_obj = json.loads(raw)
            content = response_obj["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            story = " ".join(str(parsed["story"]).split()).strip()
            if not story:
                raise ValueError("empty_story")
            return story
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
        except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError(f"Synthetic generation failed after {max_retries} attempts: {last_error}") from last_error


def load_existing_state(path: Path) -> tuple[set[int], int]:
    seen_indices = set()
    accepted_count = 0
    if not path.exists():
        return seen_indices, accepted_count
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            seen_indices.add(int(record["source_index"]))
            if record.get("accepted"):
                accepted_count += 1
    return seen_indices, accepted_count


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    dataset = load_dataset(args.source_dataset, split=args.source_split)
    dataset = dataset.shuffle(seed=args.seed)
    if args.source_limit:
        dataset = dataset.select(range(min(args.source_limit, len(dataset))))

    accepted_path = Path(args.accepted_jsonl)
    rejected_path = Path(args.rejected_jsonl)

    seen_indices = set()
    accepted_count = 0
    if args.resume:
        seen_a, accepted_count = load_existing_state(accepted_path)
        seen_r, _ = load_existing_state(rejected_path)
        seen_indices = seen_a | seen_r
        print(f"Resume mode: {accepted_count} accepted stories already present, {len(seen_indices)} source rows processed.")

    processed = 0
    rejected = 0
    random.seed(args.seed)

    for source_index, example in enumerate(dataset):
        if accepted_count >= args.target_count:
            break
        if source_index in seen_indices:
            continue

        source_text = " ".join(str(example[args.source_text_key]).split()).strip()
        if not source_text:
            continue
        source_text = source_text[: args.max_source_chars]

        try:
            synthetic_story = generate_one_story(
                source_story=source_text,
                system_prompt=prompt_text,
                model_name=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            accepted, reason, sentence_count, token_count = validate_story(synthetic_story)
            record = {
                "source_index": source_index,
                "source_story": source_text,
                "story": synthetic_story,
                "accepted": accepted,
                "reason": reason,
                "sentence_count": sentence_count,
                "token_count": token_count,
                "model": args.model,
            }
            if accepted:
                append_jsonl(accepted_path, record)
                accepted_count += 1
            else:
                append_jsonl(rejected_path, record)
                rejected += 1
        except Exception as exc:  # noqa: BLE001
            record = {
                "source_index": source_index,
                "source_story": source_text,
                "story": "",
                "accepted": False,
                "reason": f"generation_error: {exc}",
                "sentence_count": 0,
                "token_count": 0,
                "model": args.model,
            }
            append_jsonl(rejected_path, record)
            rejected += 1

        processed += 1
        if processed % 25 == 0:
            print(
                f"processed={processed}, accepted={accepted_count}, rejected={rejected}, "
                f"target={args.target_count}"
            )
        time.sleep(args.sleep)

    print("Synthetic generation completed.")
    print(f"accepted_jsonl : {accepted_path}")
    print(f"rejected_jsonl : {rejected_path}")
    print(f"accepted_count : {accepted_count}")
    print(f"processed_count: {processed}")


if __name__ == "__main__":
    main()
