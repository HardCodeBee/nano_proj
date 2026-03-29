# Frozen Baseline: `r19`

This note freezes the Task 2 comparison baseline to the best documented Task 1
public-test run in the workspace.

## Identity

- Run directory: `out-rocstories-remote-r19/`
- Frozen config: `config/train_rocstories_r19_frozen.py`
- Source recipe family: `config/train_rocstories_task1_push_v7.py`
- Architecture: official nanoGPT baby GPT
- Parameter count: `29.94M`
- Seed: `2027`

## Training Hyperparameters

- `dataset = rocstories`
- `batch_size = 80`
- `block_size = 96`
- `dropout = 0.14`
- `learning_rate = 3.5e-4`
- `weight_decay = 0.07`
- `warmup_iters = 500`
- `max_iters = 12000`
- `lr_decay_iters = 12000`
- `min_lr = 1e-5`
- `beta2 = 0.99`
- `compile = False`

## Frozen Evaluation Record

From `out-rocstories-remote-r19/eval_test_full.log`:

- Paragraphs used: `19633`
- Predicted tokens: `988345`
- Average loss: `3.216`
- Perplexity: `24.93`

## Frozen Sampling Defaults

From `out-rocstories-remote-r19/sample_params.json`:

- `temperature = 0.7`
- `top_k = 40`

## Task 2 Rule

All Task 2 experiments should compare against this baseline.
Do not overwrite the frozen config. Create a new config for every exploration run.
