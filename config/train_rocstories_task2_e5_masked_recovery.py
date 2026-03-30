"""
Task 2 E5 Stage 2b: short real-ROCStories recovery that keeps the post-EOT mask on.

Hypothesis:
- E5 Stage 1 may have learned a judge-favored continuation style.
- The original E5 recovery likely washed too much of that signal away because it
  returned to plain mixed LM training with `mask_after_story_end = False`.
- This variant keeps the real-data recovery short and conservative, but preserves
  the story-boundary cleanup that was the clearest win in E4.
"""

out_dir = "out-task2-e5-masked-recovery"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e5-synth-continuation-weighted/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e5-masked-recovery"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.6
opening_window_tokens = 24
loss_mode = "standard"
mask_after_story_end = True

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 2.5e-5
weight_decay = 7e-2
max_iters = 16000
lr_decay_iters = 16000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 25

compile = False
seed = 2027
