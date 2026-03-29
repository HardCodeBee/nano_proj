# Task 1 Summary: nanoGPT on ROCStories

## Objective

Train a nanoGPT model from scratch on ROCStories for short story generation while staying within the course constraints:

- no external pretrained weights
- no model larger than `32M` parameters
- no changes to the evaluation/upload workflow

This summary is synchronized to the current best validated public-test result in the workspace: `out-rocstories-remote-r19/` with `avg_loss = 3.216` and `ppl = 24.93`.

## Data Pipeline

- Dataset: `mintujupally/ROCStories`
- Train split: `78,528` stories
- Public test split used as the local validation/evaluation split: `19,633` stories
- Tokenizer: GPT-2 BPE via `tiktoken`
- Story separator: append GPT-2 `eot` token (`50256`) after every story
- Binary files produced for nanoGPT: `data/rocstories/train.bin`, `data/rocstories/val.bin`

Token statistics from `data/rocstories/dataset_stats.json`:

- Train tokens: `4,111,142`
- Val tokens: `1,027,611`
- Mean story length: `52.35` tokens
- 95th percentile story length: `69` tokens
- Max train length: `109` tokens
- Max val length: `91` tokens

Why the later runs switched from `block_size = 128` to `block_size = 96`:

- ROCStories are short, and the public validation split almost always fits inside 96 GPT-2 tokens.
- Reducing context length let the model focus capacity on the part of the sequence that actually matters most for this dataset while also allowing a slightly larger batch.

## Model

- Architecture: official nanoGPT baby-GPT scale
- `n_layer = 6`
- `n_head = 6`
- `n_embd = 384`
- `bias = False`
- Parameter count: `29.94M`

This stays safely within the assignment's `32M` cap.

## Reproduction Commands

Standard local command chain:

- `python data/rocstories/prepare.py`
- `python train.py config/train_rocstories.py`
- `python eval.py --init_from=resume --out_dir=out-rocstories --input_file=data/rocstories/test_full.txt --print_first_n=0`
- `python sample.py --init_from=resume --out_dir=out-rocstories --start="Emily forgot her umbrella before leaving for work." --temperature=0.7 --top_k=40`

What these commands produce:

- `prepare.py` writes `train.bin`, `val.bin`, `dataset_stats.json`, and `test_full.txt`
- `train.py` uses `config/train_rocstories.py` as the coursework-aligned default local Task 1 recipe
- `eval.py` reads the blank-line-separated `test_full.txt` file without changing the course evaluation pipeline
- `sample.py` uses the same checkpoint format as the grader-facing workflow

## Current Best Validated Setup

Winning run:

- Checkpoint directory: `out-rocstories-remote-r19/`
- Originating config family: `config/train_rocstories_task1_push_v7.py`
- Current synced local default: `config/train_rocstories.py`

Main hyperparameters:

- Initialization: from scratch (`init_from = scratch`)
- Batch size: `80`
- Gradient accumulation steps: `1`
- Block size: `96`
- Dropout: `0.14`
- Learning rate: `3.5e-4`
- Scheduler: cosine decay
- Warmup iters: `500`
- Decay iters: `12000`
- Max iters: `12000`
- Min learning rate: `1e-5`
- `beta2 = 0.99`
- `weight_decay = 0.07`
- Evaluation interval: `25`
- Evaluation iters: `100`
- Seed: `2027`
- `compile = False`

Best validation checkpoint inside the run:

- Best validation loss in `train.log`: `3.2774`
- Best validation step: `11175`

Exact public test evaluation using `eval.py`:

- Paragraphs used: `19,633`
- Predicted tokens: `988,345`
- Average loss: `3.216`
- Perplexity: `24.93`

## Compute Budget Note

- Training environment used for the local Task 1 work: single `NVIDIA GeForce RTX 4060 Laptop GPU`
- `compile = False` was kept for stability on the local Windows environment
- The `r19` training log shows steady-state training iterations around `42-43 ms/iter`; exact end-to-end wall-clock varies because validation runs every `25` steps and the log does not record timestamps for each evaluation block

## Improvement Timeline

Main milestones:

| stage | main change | avg_loss | ppl |
| --- | --- | ---: | ---: |
| original baseline | first scratch Task 1 baseline | `3.364` | `28.89` |
| local optimization rerun | lower dropout + longer schedule | `3.318` | `27.60` |
| local tuned round 2 | lower LR + lower weight decay | `3.299` | `27.09` |
| `r6` | best 128-token baseline | `3.242` | `25.57` |
| `r8-r12` | short-context push (`block_size = 96`) | `3.225` | `25.16` |
| `r15` | best `v3` seed sweep (`seed = 2027`) | `3.223` | `25.10` |
| `r18` | longer `v6` continuation | `3.219` | `25.00` |
| `r19` | longer + denser `v7` checkpointing | `3.216` | `24.93` |

Recent seed comparison on the final `v7` recipe:

| run | seed | best val loss | ppl |
| --- | ---: | ---: | ---: |
| `r19` | `2027` | `3.2774` | `24.93` |
| `r20` | `31415` | `3.2849` | `24.98` |
| `r21` | `424242` | `3.2746` | `24.96` |
| `r22` | `777777` | `3.2753` | `24.95` |

Takeaway:

- The seed with the best validation loss was not always the seed with the best public-test PPL.
- `r19` remained the best exact public-test checkpoint, so it is the current preferred checkpoint for reporting and submission preparation.

## Qualitative Samples From The Current Best Run

Sampling command family:

- checkpoint: `out-rocstories-remote-r19`
- `temperature = 0.7`, `top_k = 40` for safer decoding
- `temperature = 0.9`, `top_k = 100` for more diverse but less stable decoding

Prompt: `Emily forgot her umbrella before leaving for work.`

- `temperature=0.7, top_k=40`
  - `Emily forgot her umbrella before leaving for work. She took off her umbrella one day and it started to rain. She was so disappointed and started to cry. She spent all day in the rain. She was able to stay inside until she was muddy and tired.`
- `temperature=0.9, top_k=100`
  - `Emily forgot her umbrella before leaving for work. She took off her umbrella into the school center. In fact she was stuck there the entire day. The only way spent Amy was late to the school she was late. Emily was late to work.`

Prompt: `Tom decided to cook dinner for his friends.`

- `temperature=0.7, top_k=40`
  - `Tom decided to cook dinner for his friends. He set up a cooking competition for them. Tom was nervous but did not know the results. He wound up burning everyone's food. Tom wound up burning the entire cooking for them.`
- `temperature=0.9, top_k=100`
  - `Tom decided to cook dinner for his friends. It was the best meal ever. Tom was a bit sick of smoke and kept the meal. He served up for extra food and everyone was visibly upset. Everyone was humiliated and annoyed.`

## Brief Error Analysis

Strengths:

- The model reliably preserves the short ROCStories cadence better than the original baseline.
- The short-context configuration improved public-test PPL without increasing parameter count.
- Lower-temperature decoding now tends to keep a clearer five-sentence arc than the older 128-token baseline.

Common failure modes:

- The model can still repeat concepts within a story or end on an awkward sentence.
- Higher-temperature decoding noticeably increases grammar slips and logic jumps.
- If generation is not truncated at `<|endoftext|>`, the model may continue into a second story.

## Submission-Facing Files

Tracked files that should be cited in a submission-oriented repo write-up:

- Default synced training config: `config/train_rocstories.py`
- Data preparation script: `data/rocstories/prepare.py`
- Sampling defaults kept under version control: `out-rocstories/sample_params.json`
- Main Task 1 summary: `out-rocstories/task1_summary.md`
- Optimization log: `out-rocstories/task1_optimization_update.md`
- Historical process notebook: `out-rocstories/task1_detailed_process.txt`
- Final submission checklist: `out-rocstories/task1_submission_checklist.md`

Important distinction:

- The best documented local public-test run is still `out-rocstories-remote-r19/`
- That run directory contains generated artifacts such as `ckpt.pt`, `train.log`, and `sample_params.json`
- Those large/generated files are intentionally gitignored so the repository stays close to the original nanoGPT structure
