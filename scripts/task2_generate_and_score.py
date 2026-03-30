"""
Unified Task 2 evaluation runner.

This script keeps the three evaluation axes fixed across experiments:
1. avg_loss / perplexity via the unmodified course eval.py path
2. generation on a fixed prompt set
3. Qwen-based automatic scoring with the shared rubric

Typical usage:
python scripts/task2_generate_and_score.py ^
  --run-name r19-baseline ^
  --out-dir out-rocstories-remote-r19 ^
  --dataset-recipe "ROCStories baby-GPT baseline (r19)"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "prompts" / "task2_eval_prompts.txt"
DEFAULT_QWEN_PROMPT_FILE = PROJECT_ROOT / "instruction" / "Qwen_scoring_prompt tha.txt"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "out-task2" / "results.csv"
DEFAULT_EVAL_INPUT = PROJECT_ROOT / "data" / "rocstories" / "val_full.txt"
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "out-task2" / "samples"
DEFAULT_EVAL_LOG_DIR = PROJECT_ROOT / "out-task2" / "eval_logs"
EOT_TOKEN_ID = 50256
TERMINAL_PUNCTUATION = (".", "!", "?", "\"", "'")
DRIFT_REASON_PATTERNS = (
    "ignore",
    "ignores",
    "ignored",
    "off prompt",
    "off-prompt",
    "opening sentence",
    "does not follow",
    "doesn't follow",
    "fails to follow",
)
FALLBACK_QWEN_SYSTEM_PROMPT = """You are a creative writing evaluator. Score the following short story on a scale of 1 to 5 using this rubric:

1 — Incoherent, highly repetitive, or completely ignores the opening sentence
2 — Loosely follows the prompt; major issues (logic gaps, abrupt cutoff, repetition)
3 — Adequate story; follows prompt and has a conclusion, but bland or has minor flaws
4 — Good story; coherent, natural language, satisfying ending
5 — Excellent; creative, engaging, all sentences connect, natural conclusion

Respond ONLY with a JSON object in this exact format (no extra text):
{"score": N, "reason": "brief reason"}

Where N is an integer from 1 to 5.
"""


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype() -> str:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    return "float32"


def default_judge_api_key() -> str | None:
    return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")


def default_judge_base_url() -> str:
    if os.getenv("QWEN_BASE_URL"):
        return os.getenv("QWEN_BASE_URL")
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    if os.getenv("OPENAI_API_KEY") and not (os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        return "https://api.openai.com/v1"
    return "https://dashscope.aliyuncs.com/compatible-mode/v1"


def default_judge_model() -> str:
    if os.getenv("QWEN_MODEL"):
        return os.getenv("QWEN_MODEL")
    if os.getenv("OPENAI_MODEL"):
        return os.getenv("OPENAI_MODEL")
    if os.getenv("OPENAI_API_KEY") and not (os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        return "gpt-4o-mini"
    return "qwen-plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Task 2 evaluation for one checkpoint.")
    parser.add_argument("--run-name", required=True, help="Short run identifier written to results.csv.")
    parser.add_argument("--out-dir", required=True, help="Checkpoint directory containing ckpt.pt.")
    parser.add_argument(
        "--dataset-recipe",
        required=True,
        help="Short description of the dataset/training recipe for this run.",
    )
    parser.add_argument(
        "--eval-input-file",
        default=str(DEFAULT_EVAL_INPUT),
        help="Defaults to ROCStories val_full.txt; pass locked_test.txt only for occasional final checks.",
    )
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE))
    parser.add_argument("--qwen-prompt-file", default=str(DEFAULT_QWEN_PROMPT_FILE))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--samples-jsonl", default=None, help="Optional per-sample output path.")
    parser.add_argument("--eval-log-file", default=None, help="Optional eval.py stdout log path.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default=default_dtype(), choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples to draw per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=None, help="Override sample_params.json temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Override sample_params.json top_k.")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip automatic Qwen scoring.")
    parser.add_argument("--qwen-model", default=default_judge_model())
    parser.add_argument(
        "--qwen-base-url",
        default=default_judge_base_url(),
        help="OpenAI-compatible base URL. Falls back to QWEN/OpenAI env vars before the DashScope default.",
    )
    parser.add_argument(
        "--qwen-api-key",
        default=default_judge_api_key(),
        help="API key. Defaults to QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY.",
    )
    parser.add_argument("--qwen-timeout", type=int, default=60)
    parser.add_argument("--qwen-max-retries", type=int, default=3)
    parser.add_argument("--qwen-sleep", type=float, default=0.2)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_qwen_system_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'SYSTEM_PROMPT\s*=\s*""".*?\n(.*?)"""', text, flags=re.DOTALL)
    if not match:
        return FALLBACK_QWEN_SYSTEM_PROMPT
    prompt = match.group(1).replace("\\\n", "\n").strip()
    return prompt if prompt else FALLBACK_QWEN_SYSTEM_PROMPT


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def score_story_with_qwen(
    story: str,
    system_prompt: str,
    model_name: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_retries: int,
) -> dict:
    url = normalize_base_url(base_url)
    payload = {
        "model": model_name,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story},
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
            score = int(parsed["score"])
            reason = str(parsed["reason"]).strip()
            if score < 1 or score > 5:
                raise ValueError(f"Qwen score out of range: {score}")
            return {"score": score, "reason": reason}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
        except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 4))

    raise RuntimeError(f"Qwen scoring failed after {max_retries} attempts: {last_error}") from last_error


def run_eval_via_course_script(
    out_dir: Path,
    eval_input_file: Path,
    device: str,
    dtype: str,
    compile_model: bool,
) -> tuple[float, float, str]:
    command = [
        sys.executable,
        "eval.py",
        "--init_from=resume",
        f"--out_dir={out_dir}",
        f"--input_file={eval_input_file}",
        "--print_first_n=0",
        f"--device={device}",
        f"--dtype={dtype}",
        f"--compile={str(compile_model)}",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout

    avg_loss_match = re.search(r"avg_loss\s*:\s*([0-9.]+)", stdout)
    ppl_match = re.search(r"ppl\s*:\s*([0-9.]+)", stdout)
    if not avg_loss_match or not ppl_match:
        raise RuntimeError("Failed to parse avg_loss/ppl from eval.py output.")

    avg_loss = float(avg_loss_match.group(1))
    ppl = float(ppl_match.group(1))
    return avg_loss, ppl, stdout


def load_model_and_tokenizer(out_dir: Path, device: str, dtype: str, compile_model: bool, seed: int):
    from model import GPT, GPTConfig

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = out_dir / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)

    load_meta = False
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        meta_path = PROJECT_ROOT / "data" / checkpoint["config"]["dataset"] / "meta.pkl"
        load_meta = meta_path.exists()
    if load_meta:
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        stoi = meta["stoi"]
        itos = meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[idx] for idx in ids)
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda ids: enc.decode(ids)

    return model, checkpoint, encode, decode, ctx


def load_sampling_params(out_dir: Path, temperature_override: float | None, top_k_override: int | None) -> tuple[float, int]:
    params_path = out_dir / "sample_params.json"
    if params_path.exists():
        sample_params = json.loads(params_path.read_text(encoding="utf-8"))
    else:
        sample_params = {"temperature": 0.8, "top_k": 200}

    temperature = float(sample_params.get("temperature", 0.8))
    top_k = int(sample_params.get("top_k", 200))
    if temperature_override is not None:
        temperature = float(temperature_override)
    if top_k_override is not None:
        top_k = int(top_k_override)
    return temperature, top_k


def detect_repetition(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if not lowered:
        return False

    sentences = [segment.strip() for segment in re.split(r"[.!?]+", lowered) if segment.strip()]
    sentence_counts = Counter(sentences)
    if any(count >= 2 for count in sentence_counts.values()):
        return True

    tokens = lowered.split()
    if len(tokens) < 8:
        return False
    ngrams = Counter(tuple(tokens[idx : idx + 4]) for idx in range(len(tokens) - 3))
    return any(count >= 2 for count in ngrams.values())


def detect_truncation(continuation_text: str, generated_tokens: int, max_new_tokens: int, ended_with_eot: bool) -> bool:
    stripped = continuation_text.rstrip()
    if not stripped:
        return generated_tokens >= max_new_tokens and not ended_with_eot
    if ended_with_eot:
        return False
    if generated_tokens < max_new_tokens and stripped.endswith(TERMINAL_PUNCTUATION):
        return False
    return generated_tokens >= max_new_tokens or not stripped.endswith(TERMINAL_PUNCTUATION)


def detect_prompt_drift(prompt: str, full_text: str, qwen_reason: str | None) -> bool:
    if qwen_reason:
        lowered_reason = qwen_reason.lower()
        if any(pattern in lowered_reason for pattern in DRIFT_REASON_PATTERNS):
            return True

    prompt_tokens = set(re.findall(r"[a-z]+", prompt.lower()))
    story_tokens = set(re.findall(r"[a-z]+", full_text.lower()))
    if not prompt_tokens:
        return False
    overlap = len(prompt_tokens & story_tokens)
    return overlap <= max(1, len(prompt_tokens) // 6)


def generate_samples(
    model,
    prompts: list[str],
    encode,
    decode,
    ctx,
    device: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[dict]:
    records = []
    with torch.no_grad():
        with ctx:
            for prompt in prompts:
                prompt_ids = encode(prompt)
                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
                for sample_idx in range(num_samples):
                    output = model.generate(
                        prompt_tensor,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )[0].tolist()
                    continuation_ids = output[len(prompt_ids) :]
                    ended_with_eot = EOT_TOKEN_ID in continuation_ids
                    if ended_with_eot:
                        continuation_ids = continuation_ids[: continuation_ids.index(EOT_TOKEN_ID)]
                    continuation_text = decode(continuation_ids)
                    full_text = decode(prompt_ids + continuation_ids)
                    records.append(
                        {
                            "prompt": prompt,
                            "sample_index": sample_idx,
                            "generated_text": full_text,
                            "continuation_text": continuation_text,
                            "generated_tokens": len(output) - len(prompt_ids),
                            "ended_with_eot": ended_with_eot,
                        }
                    )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def upsert_results_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_name",
        "out_dir",
        "dataset_recipe",
        "avg_loss",
        "ppl",
        "mean_qwen_score",
        "qwen_scored_samples",
        "score_1_count",
        "score_2_count",
        "score_3_count",
        "score_4_count",
        "score_5_count",
        "repetition_failures",
        "truncation_failures",
        "prompt_drift_failures",
        "temperature",
        "top_k",
        "max_new_tokens",
        "prompt_count",
        "prompts_file",
        "samples_file",
    ]

    rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("run_name") == row["run_name"]:
            rows[idx] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    prompts_file = Path(args.prompts_file)
    qwen_prompt_file = Path(args.qwen_prompt_file)
    results_csv = Path(args.results_csv)
    eval_input_file = Path(args.eval_input_file)

    if not out_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}")
    if not (out_dir / "ckpt.pt").exists():
        raise FileNotFoundError(f"Missing ckpt.pt in {out_dir}")

    prompts = load_prompts(prompts_file)
    qwen_system_prompt = load_qwen_system_prompt(qwen_prompt_file)

    if args.samples_jsonl:
        samples_jsonl = Path(args.samples_jsonl)
    else:
        samples_jsonl = DEFAULT_SAMPLES_DIR / f"{args.run_name}.jsonl"

    if args.eval_log_file:
        eval_log_file = Path(args.eval_log_file)
    else:
        eval_log_file = DEFAULT_EVAL_LOG_DIR / f"{args.run_name}_eval.txt"

    avg_loss, ppl, eval_stdout = run_eval_via_course_script(
        out_dir=out_dir,
        eval_input_file=eval_input_file,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
    )
    eval_log_file.parent.mkdir(parents=True, exist_ok=True)
    eval_log_file.write_text(eval_stdout, encoding="utf-8")

    model, checkpoint, encode, decode, ctx = load_model_and_tokenizer(
        out_dir=out_dir,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        seed=args.seed,
    )
    temperature, top_k = load_sampling_params(out_dir, args.temperature, args.top_k)

    samples = generate_samples(
        model=model,
        prompts=prompts,
        encode=encode,
        decode=decode,
        ctx=ctx,
        device=args.device,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    score_counter = Counter()
    scored_samples = 0
    total_score = 0.0
    repetition_failures = 0
    truncation_failures = 0
    prompt_drift_failures = 0

    if not args.skip_qwen and not args.qwen_api_key:
        raise ValueError(
            "Qwen scoring is enabled but no API key was provided. "
            "Set QWEN_API_KEY, DASHSCOPE_API_KEY, OPENAI_API_KEY, or pass --qwen-api-key."
        )

    for record in samples:
        repetition_failure = detect_repetition(record["continuation_text"])
        truncation_failure = detect_truncation(
            continuation_text=record["continuation_text"],
            generated_tokens=record["generated_tokens"],
            max_new_tokens=args.max_new_tokens,
            ended_with_eot=record["ended_with_eot"],
        )
        qwen_score = None
        qwen_reason = None
        if not args.skip_qwen:
            scored = score_story_with_qwen(
                story=record["generated_text"],
                system_prompt=qwen_system_prompt,
                model_name=args.qwen_model,
                base_url=args.qwen_base_url,
                api_key=args.qwen_api_key,
                timeout=args.qwen_timeout,
                max_retries=args.qwen_max_retries,
            )
            qwen_score = scored["score"]
            qwen_reason = scored["reason"]
            score_counter[qwen_score] += 1
            total_score += qwen_score
            scored_samples += 1
            time.sleep(args.qwen_sleep)

        prompt_drift_failure = detect_prompt_drift(
            prompt=record["prompt"],
            full_text=record["generated_text"],
            qwen_reason=qwen_reason,
        )

        repetition_failures += int(repetition_failure)
        truncation_failures += int(truncation_failure)
        prompt_drift_failures += int(prompt_drift_failure)

        record["temperature"] = temperature
        record["top_k"] = top_k
        record["max_new_tokens"] = args.max_new_tokens
        record["repetition_failure"] = repetition_failure
        record["truncation_failure"] = truncation_failure
        record["prompt_drift_failure"] = prompt_drift_failure
        record["qwen_score"] = qwen_score
        record["qwen_reason"] = qwen_reason

    write_jsonl(samples_jsonl, samples)

    mean_qwen_score = ""
    if scored_samples:
        mean_qwen_score = f"{(total_score / scored_samples):.4f}"

    dataset_name = "unknown"
    if "config" in checkpoint and "dataset" in checkpoint["config"]:
        dataset_name = str(checkpoint["config"]["dataset"])

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "dataset_recipe": args.dataset_recipe,
        "avg_loss": f"{avg_loss:.4f}",
        "ppl": f"{ppl:.4f}",
        "mean_qwen_score": mean_qwen_score,
        "qwen_scored_samples": scored_samples,
        "score_1_count": score_counter.get(1, 0),
        "score_2_count": score_counter.get(2, 0),
        "score_3_count": score_counter.get(3, 0),
        "score_4_count": score_counter.get(4, 0),
        "score_5_count": score_counter.get(5, 0),
        "repetition_failures": repetition_failures,
        "truncation_failures": truncation_failures,
        "prompt_drift_failures": prompt_drift_failures,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(prompts),
        "prompts_file": str(prompts_file),
        "samples_file": str(samples_jsonl),
    }
    upsert_results_row(results_csv, row)

    print("Task 2 evaluation completed.")
    print(f"run_name            : {args.run_name}")
    print(f"dataset             : {dataset_name}")
    print(f"dataset_recipe      : {args.dataset_recipe}")
    print(f"avg_loss            : {avg_loss:.4f}")
    print(f"ppl                 : {ppl:.4f}")
    print(f"mean_qwen_score     : {mean_qwen_score if mean_qwen_score else 'skipped'}")
    print(f"repetition_failures : {repetition_failures}")
    print(f"truncation_failures : {truncation_failures}")
    print(f"prompt_drift_failures: {prompt_drift_failures}")
    print(f"samples_file        : {samples_jsonl}")
    print(f"results_csv         : {results_csv}")


if __name__ == "__main__":
    main()
