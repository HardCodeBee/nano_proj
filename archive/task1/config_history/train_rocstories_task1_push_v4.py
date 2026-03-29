"""
Fourth ROCStories Task 1 push after block_size=96 improved public-test PPL to 25.16.

This keeps the shorter context that finally moved the metric, then slightly
strengthens regularization and lengthens the smoother decay window.
"""

out_dir = "out-rocstories-task1-push-v4"
eval_interval = 100
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v4"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.16

learning_rate = 3.2e-4
weight_decay = 8e-2
max_iters = 10400
lr_decay_iters = 10400
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 600

compile = False
seed = 1337
