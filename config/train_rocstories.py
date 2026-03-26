"""
This config defines the Task 1 ROCStories training setup.
It keeps the model at official nanoGPT baby-GPT scale so the run stays within the
assignment's 32M parameter limit, while using a longer scratch-training schedule and
lighter dropout than the initial baseline.
"""

out_dir = "out-rocstories"
eval_interval = 100
eval_iters = 100
log_interval = 10

# Save only the currently best checkpoint instead of keeping every evaluation snapshot.
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories"

dataset = "rocstories"
# On this machine, one larger batch was faster than using gradient accumulation.
gradient_accumulation_steps = 1
batch_size = 64
# ROCStories are short, so 128 tokens comfortably covers nearly every full story.
block_size = 128

# Official nanoGPT "baby GPT" scale.
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# Tuned Task 1 follow-up: keep the current best validated settings and extend
# training to test whether later checkpoints can push perplexity below 25.70.
learning_rate = 4e-4
weight_decay = 5e-2
max_iters = 10000
lr_decay_iters = 10000
min_lr = 4e-5
beta2 = 0.99
warmup_iters = 400

# More predictable than torch.compile() for this Windows + laptop GPU environment.
compile = False
