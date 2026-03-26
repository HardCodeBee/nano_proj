"""
This config defines the current Task 1 ROCStories training setup.
It keeps the model at official nanoGPT baby-GPT scale so the run stays within the
assignment's 32M parameter limit, while using the tuned schedule selected after the
first optimization rerun.
"""

out_dir = "out-rocstories-minlr4e5"
eval_interval = 200
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

# Tuned Task 1 schedule: keep the current best main settings and lower only min_lr
# to test whether slightly finer late-stage convergence can reduce perplexity further.
learning_rate = 4e-4
weight_decay = 5e-2
max_iters = 8000
lr_decay_iters = 8000
min_lr = 4e-5
beta2 = 0.99
warmup_iters = 300

# More predictable than torch.compile() for this Windows + laptop GPU environment.
compile = False
