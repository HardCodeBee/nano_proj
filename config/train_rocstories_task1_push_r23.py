"""
ROCStories Task 1 follow-up r23.

This is the lowest-risk continuation from the current best `v7/r19` recipe:
- keep the short-context setup that reached 24.93 PPL
- keep the stronger seed 2027
- lower only the cosine floor to test a gentler late-stage finish
"""

out_dir = "out-rocstories-remote-r23"
eval_interval = 25
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r23"

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
min_lr = 5e-6
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
