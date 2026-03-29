"""
Default ROCStories training config for coursework-compliant local reproduction.

This file is synchronized to the current best validated public-test run in the repo:
`config/train_rocstories_task1_push_v7.py` -> `out-rocstories-remote-r19/`
with `avg_loss = 3.216` and `ppl = 24.93`.

It keeps the official nanoGPT baby-GPT architecture, stays under the assignment's
32M parameter limit, and trains entirely from scratch.
"""

out_dir = "out-rocstories"
eval_interval = 25
eval_iters = 100
log_interval = 10

# Save only the currently best checkpoint instead of keeping every evaluation snapshot.
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
# Public ROCStories stories are short, so a 96-token context matches the data well.
block_size = 96

# Official nanoGPT "baby GPT" scale.
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

# Current best validated run matches the longer v7 push recipe.
learning_rate = 3.5e-4
weight_decay = 7e-2
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

# More predictable than torch.compile() for this Windows + laptop GPU environment.
compile = False
seed = 2027
