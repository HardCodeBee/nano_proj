# TinyStories -> ROCStories Curriculum Plan

## Why this direction is reasonable

- TinyStories is still a story-generation dataset, so it stays aligned with the clarified Task 2 scope.
- Its stories are cleaner and more locally coherent than many longer web-story datasets.
- It can be used as Stage 1 curriculum data, then ROCStories can pull the model back to the exact target style in Stage 2.

## Key data observation

Prepared subset:

- train stories: `200,000`
- val stories: `5,000`
- train tokens: `44,809,105`
- val tokens: `1,085,200`
- train mean tokens/story: `224.05`
- train p95 tokens/story: `457`

Compared with ROCStories, TinyStories is much longer, but the first experiment keeps
`block_size = 96` on purpose so the curriculum checkpoint can resume directly into the
existing ROCStories setup.

## E1 experiment

### Stage 1

- Config: `config/train_tinystories_task2_e1_stage1.py`
- Dataset: TinyStories subset
- Objective: learn cleaner short-range narrative transitions and endings
- Training budget: `4000` steps

### Stage 2

- Config: `config/train_rocstories_task2_e1_stage2.py`
- Dataset: ROCStories
- Resume from the Stage 1 out_dir
- Objective: pull the model back to the course target domain
- Total budget after resume: `12000` steps

## Comparison rule

Compare E1 against the frozen baseline:

- Baseline: `r19`
- Config reference: `config/train_rocstories_r19_frozen.py`
- Daily metrics: `ROC val` `avg_loss`, `ppl`, fixed-prompt samples, and automatic judge score
- Locked local check: `ROC locked_test` only for shortlisted checkpoints

## Commands

### Prepare TinyStories subset

```bash
./.venv/bin/python data/tinystories/prepare.py \
  --max-train-examples 200000 \
  --max-val-examples 5000
```

### Stage 1

```bash
./.venv/bin/python train.py config/train_tinystories_task2_e1_stage1.py
```

### Stage 2

```bash
./.venv/bin/python train.py config/train_rocstories_task2_e1_stage2.py
```

### Fixed evaluation

```bash
./.venv/bin/python scripts/task2_generate_and_score.py \
  --run-name e1-tinystories-to-rocstories \
  --out-dir out-task2-e1-tinystories-to-rocstories \
  --dataset-recipe "TinyStories subset -> ROCStories curriculum"
```
