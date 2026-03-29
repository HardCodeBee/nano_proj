"""
ROCStories Task 1 follow-up r31.

Purpose:
- combine the two strongest remaining ideas after r26
- extend the tail modestly
- evaluate densely and with higher precision throughout the run
"""

out_dir = "out-rocstories-remote-r31"
eval_interval = 5
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r31"

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
