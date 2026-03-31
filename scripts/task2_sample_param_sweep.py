"""
Sweep Task 2 decoding parameters with the fixed scorer and optionally install the
best sample_params.json for submission-safe generation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "out-task2" / "results.csv"
SCORER_SCRIPT = PROJECT_ROOT / "scripts" / "task2_generate_and_score.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Task 2 sample_params.json decoding settings.")
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-recipe", required=True)
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--qwen-prompt-file", default=None)
    parser.add_argument("--eval-input-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-timeout", type=int, default=None)
    parser.add_argument("--qwen-max-retries", type=int, default=None)
    parser.add_argument("--qwen-sleep", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[40, 80, 120])
    parser.add_argument("--write-best-sample-params", action="store_true")
    parser.add_argument("--sample-params-output", default=None)
    return parser.parse_args()


def format_temperature_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def build_run_name(prefix: str, temperature: float, top_k: int) -> str:
    return f"{prefix}-t{format_temperature_tag(temperature)}-k{top_k}"


def maybe_extend_arg(command: list[str], flag: str, value) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def load_result_row(results_csv: Path, run_name: str) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    matched_rows = []
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("run_name") == run_name:
                matched_rows.append(row)

    if not matched_rows:
        raise ValueError(f"No results row found for run_name={run_name} in {results_csv}")
    return matched_rows[-1]


def parse_float(value: str | None, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def parse_int(value: str | None, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def ranking_key(row: dict) -> tuple[float, int, float]:
    mean_score = parse_float(row.get("mean_qwen_score"), -1.0)
    failures = (
        parse_int(row.get("repetition_failures"), 0)
        + parse_int(row.get("truncation_failures"), 0)
        + parse_int(row.get("prompt_drift_failures"), 0)
    )
    temperature = parse_float(row.get("temperature"), 0.0)
    return (mean_score, -failures, -temperature)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    results_csv = Path(args.results_csv)
    best_row = None

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
            "--results-csv",
            str(results_csv),
            "--num-samples",
            str(args.num_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
        ]
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

        print(f"[sweep] running {run_name} with temperature={temperature}, top_k={top_k}")
        subprocess.run(command, check=True)
        row = load_result_row(results_csv, run_name)
        if best_row is None or ranking_key(row) > ranking_key(best_row):
            best_row = row

    if best_row is None:
        raise RuntimeError("Sweep completed without any result rows.")

    best_temperature = parse_float(best_row.get("temperature"), 0.8)
    best_top_k = parse_int(best_row.get("top_k"), 200)
    print(
        "[sweep] best setting: "
        f"run_name={best_row['run_name']}, mean_qwen_score={best_row.get('mean_qwen_score', '')}, "
        f"temperature={best_temperature}, top_k={best_top_k}"
    )

    if args.write_best_sample_params:
        sample_params_path = Path(args.sample_params_output) if args.sample_params_output else out_dir / "sample_params.json"
        sample_params_path.parent.mkdir(parents=True, exist_ok=True)
        sample_params_path.write_text(
            json.dumps({"temperature": best_temperature, "top_k": best_top_k}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[sweep] wrote sample params to {sample_params_path}")


if __name__ == "__main__":
    main()
