"""
ROCStories Task 1 follow-up r29.

Purpose:
- build directly on the successful r26 checkpoint-selection strategy
- keep the same dense/precise validation settings
- extend the tail slightly in case the improved selection window continues to help
"""

out_dir = "out-rocstories-remote-r29"
eval_interval = 10
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r29"

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
max_iters = 12400
lr_decay_iters = 12400
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
