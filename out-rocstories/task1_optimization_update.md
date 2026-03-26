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

The second tuning round was the current best local Task 1 result before the remote validation run.

## Prepared R1 Follow-Up

The next prepared Task 1 configuration keeps the current best main settings and changes only:

- `min_lr: 6e-5 -> 4e-5`

All other main hyperparameters remain:

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `beta2 = 0.99`

## Remote Validation Run

After the local tuning rounds, the best-performing main settings were run on the rented remote GPU and evaluated exactly on the full public test text file.

- Validated hyperparameters:
  - `dropout = 0.1`
  - `learning_rate = 4e-4`
  - `weight_decay = 0.05`
  - `max_iters = 8000`
  - `lr_decay_iters = 8000`
  - `warmup_iters = 300`
  - `min_lr = 6e-5`
  - `beta2 = 0.99`
- Remote device:
  - `NVIDIA RTX A6000`
- Best checkpoint metadata:
  - `iter_num = 8000`
  - `best_val_loss = 3.2865`
- Exact public test evaluation:
  - `avg_loss = 3.253`
  - `ppl = 25.86`

## Current Best Task 1 Result

- Original baseline:
  - `ppl = 28.89`
- First optimization rerun:
  - `ppl = 27.60`
- Second local tuning round:
  - `ppl = 27.09`
- Remote validated result:
  - `ppl = 25.86`

The remote validated run is now the current best Task 1 result in this workspace.

## Remote `min_lr = 4e-5` Follow-Up

The next remote follow-up changed only one main hyperparameter relative to the remote validated result:

- `min_lr: 6e-5 -> 4e-5`

All other main settings were kept the same:

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `beta2 = 0.99`

Exact public test evaluation on the remote follow-up:

- `avg_loss = 3.247`
- `ppl = 25.70`

This became the new best validated Task 1 result.

## Remote 10k-Step Follow-Up

After the `min_lr = 4e-5` improvement, a longer remote continuation was tested to see whether simply extending training would help further.

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 10000`
- `lr_decay_iters = 10000`
- `warmup_iters = 400`
- `min_lr = 4e-5`
- `beta2 = 0.99`

Best checkpoint metadata from the copied-back snapshot:

- `iter_num = 9700`
- `best_val_loss = 3.2957`

Exact public test evaluation:

- `avg_loss = 3.255`
- `ppl = 25.92`

This did not beat the 8k-step `min_lr = 4e-5` run, so longer training was not kept as the primary direction.

## Remote Checkpoint-Selection Follow-Up

After the 10k-step continuation underperformed, a smaller follow-up tested whether steadier validation estimates and a smoother AdamW second-moment setting would help checkpoint selection near the 8k-step best region.

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `min_lr = 4e-5`
- `beta2 = 0.995`
- `eval_iters = 200`

Best checkpoint metadata from the copied-back snapshot:

- `iter_num = 7900`
- `best_val_loss = 3.2935`

Exact public test evaluation:

- `avg_loss = 3.251`
- `ppl = 25.83`

This beat the earlier `25.86` and `25.92` remote follow-ups, but it still did not surpass the current best `25.70`.

## Remote `min_lr = 3e-5` Follow-Up (`r4`)

After the smoother-checkpoint-selection run still fell short of the best 8k-step result, one more remote follow-up tested whether a slightly lower cosine floor could improve the final stretch without changing the rest of the recipe.

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `min_lr = 3e-5`
- `beta2 = 0.99`
- `eval_iters = 100`

Best checkpoint metadata from the copied-back snapshot:

- `iter_num = 8000`
- `best_val_loss = 3.2789`

Exact public test evaluation:

- `avg_loss = 3.244`
- `ppl = 25.65`

This became the new best validated Task 1 result.

## Remote `min_lr = 2e-5` Follow-Up (`r5`)

After the `r4` run improved again, one more follow-up tested whether lowering the cosine floor from `3e-5` to `2e-5` could still help the final stage of the same 8k-step recipe.

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `min_lr = 2e-5`
- `beta2 = 0.99`
- `eval_iters = 100`

Best checkpoint metadata from the copied-back snapshot:

- `iter_num = 8000`
- `best_val_loss = 3.2774`

Exact public test evaluation:

- `avg_loss = 3.242`
- `ppl = 25.59`

This became the new best validated Task 1 result.

## Remote `min_lr = 1e-5` Follow-Up (`r6`)

After the `r5` improvement, one final follow-up pushed the cosine floor down once more to check whether the same 8k-step recipe still benefited from an even smaller ending learning rate.

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `min_lr = 1e-5`
- `beta2 = 0.99`
- `eval_iters = 100`

Best checkpoint metadata from the copied-back snapshot:

- `iter_num = 8000`
- `best_val_loss = 3.2763`

Exact public test evaluation:

- `avg_loss = 3.242`
- `ppl = 25.57`

This became the new best validated Task 1 result.

## Updated Best Result Table

- Original baseline:
  - `ppl = 28.89`
- First optimization rerun:
  - `ppl = 27.60`
- Second local tuning round:
  - `ppl = 27.09`
- Remote validated result (`min_lr = 6e-5`):
  - `ppl = 25.86`
- Remote best result (`min_lr = 4e-5`):
  - `ppl = 25.70`
- Remote 10k-step follow-up:
  - `ppl = 25.92`
- Remote checkpoint-selection follow-up:
  - `ppl = 25.83`
- Remote follow-up (`min_lr = 3e-5`):
  - `ppl = 25.65`
- Remote follow-up (`min_lr = 2e-5`):
  - `ppl = 25.59`
- Remote latest best result (`min_lr = 1e-5`):
  - `ppl = 25.57`

The current best validated Task 1 result is now `ppl = 25.57`.

## Checked-In Config Restored To Best State

The checked-in local `config/train_rocstories.py` has now been restored to the current best validated Task 1 setup:

- `dropout = 0.1`
- `learning_rate = 4e-4`
- `weight_decay = 0.05`
- `max_iters = 8000`
- `lr_decay_iters = 8000`
- `warmup_iters = 300`
- `min_lr = 1e-5`
- `beta2 = 0.99`
- `eval_iters = 100`

This keeps the checked-in training config aligned with the exact best public-test result of `ppl = 25.57`.
