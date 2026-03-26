/**
 * File role:
 * This is the short, report-friendly summary of the optimized Task 1 run.
 * Read this file when you want the final setup, refreshed metrics, and updated
 * sample outputs quickly.
 */

# Task 1 Summary: nanoGPT on ROCStories

## Objective

Train a nanoGPT model from scratch on ROCStories for short story generation, staying within the 32M parameter budget. This summary reflects the optimized Task 1 rerun that used a longer scratch-training schedule and lighter dropout than the original baseline.

## Data Pipeline

- Dataset: `mintujupally/ROCStories`
- Train split: 78,528 stories
- Public test split used as local validation/evaluation split: 19,633 stories
- Tokenizer: GPT-2 BPE via `tiktoken`
- Story boundary handling: append GPT-2 `eot` token (`50256`) after every story
- Binary files generated for nanoGPT: `data/rocstories/train.bin`, `data/rocstories/val.bin`

Token statistics from `data/rocstories/dataset_stats.json`:

- Train tokens: 4,111,142
- Val tokens: 1,027,611
- Mean story length: 52.35 tokens
- 95th percentile story length: 69 tokens
- Max story length: 109 tokens

Based on these statistics, `block_size=128` was kept because one full story still fits comfortably in context.

## Model

- Architecture: official nanoGPT baby GPT scale
- `n_layer = 6`
- `n_head = 6`
- `n_embd = 384`
- `dropout = 0.1`
- `bias = False`
- Parameter count: 29.94M

## Training Setup

- Config: `config/train_rocstories.py`
- Initialization: from scratch (`init_from = scratch`)
- Device: single GPU (`NVIDIA GeForce RTX 4060 Laptop GPU`)
- Batch size: 64
- Gradient accumulation steps: 1
- Effective tokens/iteration: 8,192
- Learning rate: 6e-4
- Scheduler: cosine decay
- Warmup iters: 400
- Decay iters: 12000
- Min learning rate: 6e-5
- Max iters: 12000
- `beta2 = 0.99`
- `weight_decay = 0.1` (inherited from `train.py`)
- Seed: 1337 (inherited from `train.py`)
- `compile = False`

Approximate wall-clock time:

- Data preparation: already available
- Training run: ~55 minutes on the RTX 4060 Laptop GPU
- Full public test evaluation: ~1.5 minutes

## Training Curve Highlights

- Step 0: val loss 10.8924
- Step 200: val loss 5.1558
- Step 1000: val loss 3.9110
- Step 1200: val loss 3.7923
- Step 3200: val loss 3.4353
- Step 4200: val loss 3.3894
- Best sampled validation estimate from checkpoint metadata: 3.3616 at step 5800

Note: the training run completed to 12000 steps, but the best checkpoint remained the one saved at step 5800.

## Final Quantitative Result

Exact evaluation on the full public ROCStories test split using `eval.py`:

- Paragraphs used: 19,633
- Predicted tokens: 988,345
- Average loss: 3.318
- Perplexity: 27.60

Improvement over the original Task 1 baseline:

- Average loss: `3.364 -> 3.318`
- Perplexity: `28.89 -> 27.60`

## Qualitative Samples

Prompt: `Emily forgot her umbrella before leaving for work.`

- `temperature=0.7, top_k=40`
  - `Emily forgot her umbrella before leaving for work. She had to sit down to get to work. When she got to work it was raining. Emily had to rush to work.`
- `temperature=0.9, top_k=100`
  - `Emily forgot her umbrella before leaving for work. She wanted to sit down. She was supposed to sit so she grabbed her umbrella. Then, she put the umbrella in her puddles. Luckily, she wasn't hurt but she was able to help her.`

Prompt: `Tom decided to cook dinner for his friends.`

- `temperature=0.7, top_k=40`
  - `Tom decided to cook dinner for his friends. He wanted to make a meal. He bought all the ingredients. Tom was in trouble for eating. He wound up burning his food.`
- `temperature=0.9, top_k=100`
  - `Tom decided to cook dinner for his friends. He wanted some spicy ones. He created elaborate scenes for the cooking class. Tom and Tom were very competitive fans. They all loved their meal.`

## Brief Error Analysis

- Strengths:
  - The model still preserves the short five-sentence ROCStories rhythm reasonably well.
  - The lower-dropout run improved validation loss and full-set perplexity without changing model size.
  - Lower-temperature decoding remains more coherent than higher-temperature decoding.
- Common failure modes:
  - The model can still emit a second story after `eot` if generation is not truncated there.
  - Some samples remain logically awkward or repetitive.
  - Higher-temperature decoding still increases inconsistency noticeably.

## Files Produced

- Checkpoint: `out-rocstories/ckpt.pt`
- Training log: `out-rocstories/train.log`
- Evaluation log: `out-rocstories/eval_optimized.log`
- Sampling defaults: `out-rocstories/sample_params.json`
- Summary: `out-rocstories/task1_summary.md`
- Optimization note: `out-rocstories/task1_optimization_update.md`
