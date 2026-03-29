"""
Seventh ROCStories Task 1 push.

This extends the near-threshold v6 run:
- same seed 2027 and same short-context recipe
- train a bit longer because the best val was still arriving near the end
- evaluate every 25 steps to capture any narrow late optimum
"""

out_dir = "out-rocstories-task1-push-v7"
eval_interval = 25
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v7"

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
