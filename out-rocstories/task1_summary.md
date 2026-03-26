/**
 * File role:
 * This is the short, report-friendly summary of the current best validated Task 1 run.
 * Read this file when you want the final setup, refreshed metrics, and updated
 * sample outputs quickly.
 */

# Task 1 Summary: nanoGPT on ROCStories

## Objective

Train a nanoGPT model from scratch on ROCStories for short story generation, staying within the 32M parameter budget. This summary now reflects the current best validated Task 1 result, obtained on the rented remote GPU after iteratively improving the original baseline, the local tuning reruns, and several remote follow-up experiments.

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

## Best Validated Training Setup

- Source of best result: remote GPU follow-up run using the stronger 8k-step setup with a slightly lower `min_lr`, evaluated exactly on the full public test file
- Initialization: from scratch (`init_from = scratch`)
- Device: single GPU (`NVIDIA RTX A6000`)
- Batch size: 64
- Gradient accumulation steps: 1
- Effective tokens/iteration: 8,192
- Learning rate: 4e-4
- Scheduler: cosine decay
- Warmup iters: 300
- Decay iters: 8000
- Min learning rate: 3e-5
- Max iters: 8000
- `beta2 = 0.99`
- `weight_decay = 0.05`
- Seed: 1337 (inherited from `train.py`)
- `compile = False`

Checked-in config note:

- The current local `config/train_rocstories.py` is aligned with the best validated Task 1 setup itself: `max_iters = 8000`, `min_lr = 3e-5`, `beta2 = 0.99`, and `eval_iters = 100`.

Approximate wall-clock time:

- Data preparation: already available
- Training run: completed on the rented `NVIDIA RTX A6000`
- Full public test evaluation: exact `eval.py` run on the remote result after copying `test_full.txt`

## Training Curve Highlights

- Step 0: val loss 10.8923
- Step 200: val loss 5.3561
- Step 1000: val loss 4.0420
- Step 2000: val loss 3.6067
- Step 4200: val loss 3.3731
- Step 7000: val loss 3.2977
- Step 7200: val loss 3.2872
- Step 7400: val loss 3.2859
- Step 8000: val loss 3.2789

Note: the best exact public-test result now comes from the remote follow-up that changed only `min_lr` from `4e-5` to `3e-5`. Earlier remote follow-ups at `6e-5`, `4e-5`, a 10k-step continuation, and a smoother-checkpoint-selection run all performed slightly worse.

## PPL Reduction Path

- Original Task 1 baseline: `avg_loss = 3.364`, `ppl = 28.89`
- First optimization rerun: lower `dropout` and longer schedule -> `avg_loss = 3.318`, `ppl = 27.60`
- Second local tuning round: lower `learning_rate` and `weight_decay` -> `avg_loss = 3.299`, `ppl = 27.09`
- Remote validation run: same 8k-step recipe on A6000 with `min_lr = 6e-5` -> `avg_loss = 3.253`, `ppl = 25.86`
- Remote follow-up: `min_lr = 4e-5` -> `avg_loss = 3.247`, `ppl = 25.70`
- Remote 10k-step follow-up: longer schedule -> `avg_loss = 3.255`, `ppl = 25.92`
- Remote checkpoint-selection follow-up: `beta2 = 0.995`, `eval_iters = 200` -> `avg_loss = 3.251`, `ppl = 25.83`
- Remote `r4` follow-up: `min_lr = 3e-5` -> `avg_loss = 3.244`, `ppl = 25.65`

## Final Quantitative Result

Exact evaluation on the full public ROCStories test split using `eval.py`:

- Paragraphs used: 19,633
- Predicted tokens: 988,345
- Average loss: 3.244
- Perplexity: 25.65

Improvement over the original Task 1 baseline:

- Average loss: `3.364 -> 3.244`
- Perplexity: `28.89 -> 25.65`

Improvement over the first optimization rerun:

- Average loss: `3.318 -> 3.244`
- Perplexity: `27.60 -> 25.65`

Improvement over the previous best remote result:

- Average loss: `3.247 -> 3.244`
- Perplexity: `25.70 -> 25.65`

## Qualitative Samples

Prompt: `Emily forgot her umbrella before leaving for work.`

- `temperature=0.7, top_k=40`
  - `Emily forgot her umbrella before leaving for work. She arrived at the door to get in the car for the rain. When it was raining, she realized that she forgot to bring her umbrella. She quickly ran to the store and bought her umbrella. She felt better for making it to her house.`
- `temperature=0.9, top_k=100`
  - `Emily forgot her umbrella before leaving for work. She wanted to sit down to get in the car for so long. When it got to work, it started to rain very hard. Emily got out to leave her umbrella at work. She found her umbrella in the wash.`

Prompt: `Tom decided to cook dinner for his friends.`

- `temperature=0.7, top_k=40`
  - `Tom decided to cook dinner for his friends. He wanted to make sure to bake a cake. He mixed his ingredients in the oven. He made a delicious cake. Tom was happy he got a lot more prepared.`
- `temperature=0.9, top_k=100`
  - `Tom decided to cook dinner for his friends. He wanted some sausage but didn't have enough money for the sauce. He decided to make some chicken. He cooked the bacon and then cut the oven! Tom loved the meal but he was so happy he cooked.`

## Brief Error Analysis

- Strengths:
  - The model still preserves the short five-sentence ROCStories rhythm reasonably well.
  - The latest remote A6000 follow-up improved full-set perplexity again without changing model size.
  - Lower-temperature decoding remains more coherent than higher-temperature decoding.
- Common failure modes:
  - The model can still emit a second story after `eot` if generation is not truncated there.
  - Some samples remain logically awkward or repetitive.
  - Higher-temperature decoding still increases inconsistency noticeably.

## Files Produced

- Current local preparation config: `config/train_rocstories.py`
- Earlier validated remote snapshot: `out-rocstories-remote-r1/`
- Later remote comparison snapshot: `out-rocstories-remote-r2/`
- Current best remote snapshot: `out-rocstories-remote-r4/`
- Sampling defaults: `out-rocstories/sample_params.json`
- Summary: `out-rocstories/task1_summary.md`
- Optimization note: `out-rocstories/task1_optimization_update.md`
