"""
Task 2 E6 Step 2: a milder continuation-weighted follow-up intended to preserve PPL while nudging prompt-conditioned continuation.
"""

out_dir = "out-task2-e6-gentle-continuation"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e6-mixed-masked/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e6-gentle-continuation"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.6
opening_window_tokens = 24
mask_after_story_end = True
loss_mode = "continuation_weighted"
prompt_weight = 1.0
continuation_weight = 1.25
ending_weight = 1.1
ending_tokens = 24

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 4e-5
weight_decay = 7e-2
max_iters = 17500
lr_decay_iters = 17500
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 25

compile = False
seed = 2027
