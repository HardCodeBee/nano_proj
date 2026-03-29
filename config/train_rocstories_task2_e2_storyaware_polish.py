"""
Task 2 E2: story-aware ROCStories polish starting from the validated E1 checkpoint.

Design:
- Resume from the completed TinyStories -> ROCStories curriculum checkpoint
- Save into a new out_dir so the earlier E1 artifact stays intact
- Use mixed story-aware sampling to bias updates toward story openings
- Keep a short, low-learning-rate ROCStories-only polish budget
"""

out_dir = "out-task2-e2-storyaware-polish"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e1-tinystories-to-rocstories/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e2-storyaware-polish"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.7
opening_window_tokens = 24

# These values will be forced to match the checkpoint on resume, but keeping them
# here documents the intended architecture and context length explicitly.
block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 1.2e-4
weight_decay = 7e-2
max_iters = 13500
lr_decay_iters = 13500
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 100

compile = False
seed = 2027
