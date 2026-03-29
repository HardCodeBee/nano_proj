"""
Historical ROCStories milestone config.

This file reproduces the earlier 128-token baseline that reached `25.57` PPL on
the public ROCStories test split (`out-rocstories-remote-r6`). It is no longer the
best exact result in the repo, but it remains a useful reference point before the
later short-context push (`r8+`) improved the metric further.
"""

out_dir = "out-rocstories-r6-best"
eval_interval = 200
eval_iters = 100
log_interval = 10

# Keep only the best validation checkpoint.
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r6-best"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 4e-4
weight_decay = 5e-2
max_iters = 8000
lr_decay_iters = 8000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 300

compile = False
seed = 1337
