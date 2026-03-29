"""
ROCStories Task 1 follow-up r26.

Purpose:
- keep the exact `v7/r19` hyperparameter recipe
- increase validation precision and checkpoint-selection granularity
- try to catch an even narrower late optimum around the known 11.1k-11.3k region
"""

out_dir = "out-rocstories-remote-r26"
eval_interval = 10
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r26"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 3.5e-4
weight_decay = 7e-2
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
