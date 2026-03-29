"""
ROCStories Task 1 follow-up r24.

This branch keeps the `v7/r19` training length and short-context recipe, but
slightly reduces regularization to test whether the current setup is a bit
over-regularized for ROCStories at the end of training.
"""

out_dir = "out-rocstories-remote-r24"
eval_interval = 25
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r24"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.13

learning_rate = 3.5e-4
weight_decay = 6e-2
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
