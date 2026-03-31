"""
storymix_v1 Stage A: from-scratch narrative pretraining on ROC + filtered TinyStories.
"""

out_dir = "out-storymix-v1-stageA"
init_from = "scratch"

eval_interval = 50
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "storymix-v1-stageA-scratch"

dataset = "storymix_v1"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "story_start"
mask_after_story_end = True
loss_mode = "standard"

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.10

learning_rate = 4e-4
weight_decay = 7e-2
max_iters = 16000
lr_decay_iters = 16000
min_lr = 4e-5
beta2 = 0.99
warmup_iters = 800

compile = False
seed = 2027
