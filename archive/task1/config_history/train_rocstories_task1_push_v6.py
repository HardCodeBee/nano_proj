"""
Sixth ROCStories Task 1 push.

This extends the strongest observed trajectory so far:
- keep the r15/v3 recipe that reached 25.10 PPL
- keep the stronger seed 2027 by default
- train longer because this run was still improving at step 10000
- evaluate every 50 steps so we can catch the best late checkpoint
"""

out_dir = "out-rocstories-task1-push-v6"
eval_interval = 50
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v6"

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
max_iters = 11200
lr_decay_iters = 11200
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
