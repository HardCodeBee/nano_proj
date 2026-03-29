"""
Task 2 E1 Stage 2: resume from TinyStories Stage 1 and adapt back to ROCStories.

Important:
- Uses the same out_dir as Stage 1
- Resumes the checkpoint and optimizer state from that directory
- Extends the total training budget to 12k steps, so this stage adds 8k more steps

This keeps the total step budget aligned with the frozen ROCStories baseline while
testing whether the TinyStories -> ROCStories curriculum helps story generation.
"""

out_dir = "out-task2-e1-tinystories-to-rocstories"
init_from = "resume"

eval_interval = 25
eval_iters = 100
log_interval = 10

# Stage 1 and Stage 2 use different validation distributions, so the resumed
# best_val_loss is not comparable. Force saving during Stage 2 so the final
# checkpoint definitely reflects ROCStories adaptation instead of the old
# TinyStories checkpoint.
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e1-stage2-rocstories"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80

# These will be forced to match the checkpoint on resume, but keeping them here
# makes the intended architecture explicit.
block_size = 96
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
