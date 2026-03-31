"""
tiny_roc_curriculum_v2 Stage A: short filtered TinyStories pretrain from scratch.
"""

out_dir = "out-tiny-roc-curriculum-v2-stageA"
init_from = "scratch"

eval_interval = 50
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "tiny-roc-curriculum-v2-stageA-tiny-short"

dataset = "tinystories_rocstyle_v2"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.10
sampling_mode = "story_start"
mask_after_story_end = True
loss_mode = "standard"

learning_rate = 3.0e-4
weight_decay = 7e-2
max_iters = 4000
lr_decay_iters = 4000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 300

compile = False
seed = 2027
