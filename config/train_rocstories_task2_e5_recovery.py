"""
Task 2 E5 Stage 2: short real-ROCStories recovery after continuation-aware synthetic polish.
"""

out_dir = "out-task2-e5-recovery"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e5-synth-continuation-weighted/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e5-recovery"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.7
opening_window_tokens = 12
loss_mode = "standard"
mask_after_story_end = False

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 3e-5
weight_decay = 7e-2
max_iters = 15400
lr_decay_iters = 15400
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 20

compile = False
seed = 2027
