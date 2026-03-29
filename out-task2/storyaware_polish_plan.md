# Story-Aware Polish Plan

## Goal

Take the current best validated E1 checkpoint and continue ROCStories training with a
story-aware sampler that starts more updates near story openings.

This is meant to improve:

- prompt following from the opening sentence
- local event continuity
- story endings

while staying close to standard next-token training.

## Code changes this plan depends on

- [train.py](C:/Users/12442/Desktop/GitHub/nano_proj/train.py)
  - adds `sampling_mode`
  - adds `story_sampling_prob`
  - adds `opening_window_tokens`
  - adds `init_from = "resume_path"`
- [prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories/prepare.py)
  - now exports `train_story_starts.npy`, `train_story_lengths.npy`
  - now exports `val_story_starts.npy`, `val_story_lengths.npy`

## New training config

- [train_rocstories_task2_e2_storyaware_polish.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e2_storyaware_polish.py)

Key settings:

- resume from `out-task2-e1-tinystories-to-rocstories/ckpt.pt`
- save to `out-task2-e2-storyaware-polish`
- `sampling_mode = mixed`
- `story_sampling_prob = 0.7`
- `opening_window_tokens = 24`
- short low-LR ROCStories-only continuation to `13500` total steps

## Remote run order

### 1. Pull latest code

```bash
git pull origin main
```

### 2. Regenerate ROCStories artifacts with story metadata

```bash
./.venv/bin/python data/rocstories/prepare.py
```

### 3. Run story-aware polish

```bash
./.venv/bin/python train.py config/train_rocstories_task2_e2_storyaware_polish.py
```

### 4. Evaluate with the fixed Task 2 runner

```bash
./.venv/bin/python scripts/task2_generate_and_score.py \
  --run-name e2-storyaware-polish \
  --out-dir out-task2-e2-storyaware-polish \
  --dataset-recipe "TinyStories subset -> ROCStories + story-aware ROC polish" \
  --skip-qwen
```

### 5. Optional OpenAI-judge proxy

```bash
./.venv/bin/python scripts/task2_generate_and_score.py \
  --run-name e2-storyaware-polish-openai \
  --out-dir out-task2-e2-storyaware-polish \
  --dataset-recipe "TinyStories subset -> ROCStories + story-aware ROC polish" \
  --qwen-api-key "$OPENAI_API_KEY" \
  --qwen-base-url "https://api.openai.com/v1" \
  --qwen-model "gpt-4o-mini"
```

## Success criteria

Compared with the current E1 result, look for:

- lower `ROC val` `avg_loss` / `ppl`
- fewer repetitive or semantically drifting samples
- higher local automatic-judge score if available

## Current outcome

The first E2 run has already been completed.

Confirmed result:

- run name: `e2-storyaware-polish-openai`
- eval input: `ROC val`
- `avg_loss = 3.176`
- `ppl = 23.95`
- OpenAI-judge proxy mean score: `2.0`
- repetition failures: `2`
- truncation failures: `0`
- prompt drift failures: `0`

Comparison against E1:

- token-level fit improved slightly (`24.07 -> 23.95` ppl)
- automatic judge score did not improve (`2.0 -> 2.0`)
- repetition failures increased slightly (`1 -> 2`)

Interpretation:

- story-aware polish is a positive but modest improvement
- it is worth keeping as the current best daily ROC-val checkpoint
- the gain is not yet large enough to claim a clear story-quality breakthrough
