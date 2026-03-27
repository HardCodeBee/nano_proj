"""
Fifth ROCStories Task 1 push.

This is intentionally very close to the stronger r12/v3 recipe. Instead of
changing the optimization regime again, it increases checkpoint granularity so
we are less likely to miss the true best step near the late-training sweet spot.
"""

out_dir = "out-rocstories-task1-push-v5"
eval_interval = 50
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v5"

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
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 1337
