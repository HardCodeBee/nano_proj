"""
Task 2 E5 Stage 1: continuation-aware synthetic-to-real backup line on ROCStories-style synthetic data.
"""

out_dir = "out-task2-e5-synth-continuation-weighted"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e2-storyaware-polish/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e5-synth-continuation-weighted"

dataset = "rocstories_synth"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "story_start"
mask_after_story_end = True
loss_mode = "continuation_weighted"
prompt_weight = 0.5
continuation_weight = 2.0
ending_weight = 1.25
ending_tokens = 16

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 8e-5
weight_decay = 7e-2
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 50

compile = False
seed = 2027
