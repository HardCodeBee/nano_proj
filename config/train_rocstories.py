"""
This config defines the next recommended Task 1 ROCStories training setup.
It keeps the model at official nanoGPT baby-GPT scale so the run stays within the
assignment's 32M parameter limit while focusing on smoother optimizer dynamics near
the current best 25.70 PPL result.
"""

out_dir = "out-rocstories"
eval_interval = 100
# Use a larger sampled validation average so checkpoint selection is less noisy.
eval_iters = 200
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

# Tuned Task 1 follow-up: return to the stronger 8k-step regime, keep min_lr=4e-5,
# and test a slightly smoother AdamW second-moment estimate.
learning_rate = 4e-4
weight_decay = 5e-2
max_iters = 8000
lr_decay_iters = 8000
min_lr = 4e-5
beta2 = 0.995
warmup_iters = 300

# More predictable than torch.compile() for this Windows + laptop GPU environment.
compile = False
