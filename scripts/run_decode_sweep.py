"""
Run a fixed decode sweep by repeatedly calling scripts/task2_generate_and_score.py.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORER_SCRIPT = PROJECT_ROOT / "scripts" / "task2_generate_and_score.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Task 2 decoding settings for a shortlist checkpoint.")
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-recipe", required=True)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--qwen-prompt-file", default=None)
    parser.add_argument("--eval-input-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.75, 0.85, 0.95])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[60, 120, 200])
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-timeout", type=int, default=None)
    parser.add_argument("--qwen-max-retries", type=int, default=None)
    parser.add_argument("--qwen-sleep", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    return parser.parse_args()


def maybe_extend_arg(command: list[str], flag: str, value) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def format_temperature_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def build_run_name(prefix: str, temperature: float, top_k: int) -> str:
    return f"{prefix}-t{format_temperature_tag(temperature)}-k{top_k}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}")

    for temperature, top_k in itertools.product(args.temperatures, args.top_ks):
        run_name = build_run_name(args.run_name_prefix, temperature, top_k)
        command = [
            sys.executable,
            str(SCORER_SCRIPT),
            "--run-name",
            run_name,
            "--out-dir",
            str(out_dir),
            "--dataset-recipe",
            args.dataset_recipe,
            "--num-samples",
            str(args.num_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
        ]
        maybe_extend_arg(command, "--results-csv", args.results_csv)
        maybe_extend_arg(command, "--prompts-file", args.prompts_file)
        maybe_extend_arg(command, "--qwen-prompt-file", args.qwen_prompt_file)
        maybe_extend_arg(command, "--eval-input-file", args.eval_input_file)
        maybe_extend_arg(command, "--device", args.device)
        maybe_extend_arg(command, "--dtype", args.dtype)
        maybe_extend_arg(command, "--seed", args.seed)
        maybe_extend_arg(command, "--qwen-model", args.qwen_model)
        maybe_extend_arg(command, "--qwen-base-url", args.qwen_base_url)
        maybe_extend_arg(command, "--qwen-api-key", args.qwen_api_key)
        maybe_extend_arg(command, "--qwen-timeout", args.qwen_timeout)
        maybe_extend_arg(command, "--qwen-max-retries", args.qwen_max_retries)
        maybe_extend_arg(command, "--qwen-sleep", args.qwen_sleep)
        if args.compile:
            command.append("--compile")
        if args.skip_qwen:
            command.append("--skip-qwen")

        print(f"[decode_sweep] {run_name}: temperature={temperature}, top_k={top_k}")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
