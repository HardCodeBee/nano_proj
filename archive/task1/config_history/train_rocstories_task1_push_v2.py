"""
Second ROCStories Task 1 push after the 25.35 PPL run.

This stays within the official baby-GPT setup and keeps the same data pipeline,
but nudges optimization toward a slightly gentler late-training regime:
- a slightly lower peak learning rate
- a bit more regularization
- a longer warmup
- denser evaluation checkpoints near the end
"""

out_dir = "out-rocstories-task1-push-v2"
eval_interval = 100
eval_iters = 100
log_interval = 10

# Keep only the best validation checkpoint.
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v2"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

# Official baby-GPT architecture, still under the 32M Task 1 cap.
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

# Start from the stronger r8 recipe, then smooth optimization slightly.
learning_rate = 3.5e-4
weight_decay = 7e-2
max_iters = 9200
lr_decay_iters = 9200
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 1337
