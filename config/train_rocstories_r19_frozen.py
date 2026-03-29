"""
Frozen Task 2 baseline for ROCStories.

This file captures the exact public-test-best Task 1 recipe associated with:
- config/train_rocstories_task1_push_v7.py
- out-rocstories-remote-r19/

Treat this config as immutable for Task 2 comparisons. New experiments should be
created as separate configs instead of editing this file.
"""

out_dir = "out-rocstories-r19-baseline"
eval_interval = 25
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-r19-frozen"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96

# Official nanoGPT baby-GPT scale, kept under the 32M coursework cap.
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
