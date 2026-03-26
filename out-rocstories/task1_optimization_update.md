# Task 1 Optimization Update

This note records the Task 1 optimization reruns requested after the initial baseline was completed.

## What Changed

- Kept the same model size and architecture:
  - `n_layer = 6`
  - `n_head = 6`
  - `n_embd = 384`
  - total parameters still `29.94M`
- Kept scratch training (`init_from = scratch`)
- Kept the same tokenizer and data pipeline
- Reduced `dropout` from `0.2` to `0.1`
- Extended the training schedule:
  - `max_iters: 6000 -> 12000`
  - `lr_decay_iters: 6000 -> 12000`
  - `warmup_iters: 200 -> 400`

## Why This Stays Within Course Requirements

- No external pretrained weights were used.
- The model stayed below the 32M parameter limit.
- The run remained a Task 1 ROCStories model trained from scratch.
- The change was purely an optimization of training strategy and regularization.

## First Optimization Rerun

- Best checkpoint metadata:
  - `iter_num = 5800`
  - `best_val_loss = 3.3616`
- Exact public test evaluation:
  - `avg_loss = 3.318`
  - `ppl = 27.60`

## Comparison With The Original Task 1 Baseline

- Original baseline:
  - `dropout = 0.2`
  - `max_iters = 6000`
  - `ppl = 28.89`
- Optimized rerun:
  - `dropout = 0.1`
  - `max_iters = 12000`
  - `ppl = 27.60`

The optimized run improved perplexity by `1.29` absolute points while keeping the same model scale and scratch-training setup.

## Second Tuning Round

After reviewing the first optimization rerun, the next Task 1 experiment was configured to favor better early-to-mid training dynamics instead of a longer tail:

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`

This second tuning round keeps the same architecture and still satisfies all course constraints.

## Second Tuning Round Outcome

- Best checkpoint metadata:
  - `iter_num = 5000`
  - `best_val_loss = 3.3411`
- Exact public test evaluation:
  - `avg_loss = 3.299`
  - `ppl = 27.09`

## Comparison Across Task 1 Runs

- Original baseline:
  - `dropout = 0.2`
  - `learning_rate = 6e-4`
  - `max_iters = 6000`
  - `weight_decay = 0.1`
  - `ppl = 28.89`
- First optimization rerun:
  - `dropout = 0.1`
  - `learning_rate = 6e-4`
  - `max_iters = 12000`
  - `weight_decay = 0.1`
  - `ppl = 27.60`
- Second tuning round:
  - `dropout = 0.1`
  - `learning_rate = 4e-4`
  - `max_iters = 8000`
  - `weight_decay = 0.05`
  - `ppl = 27.09`

The second tuning round is the current best Task 1 result in this workspace.
