"""
Task 2 E8 masked annealing Stage 1: move from story_start to a very gentle mixed setting.
"""

out_dir = "out-task2-e8-masked-anneal"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e4-posteot-mask/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e8-masked-anneal-stage1"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.9
opening_window_tokens = 24
mask_after_story_end = True
loss_mode = "standard"

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 6e-5
weight_decay = 7e-2
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 25

compile = False
seed = 2027
