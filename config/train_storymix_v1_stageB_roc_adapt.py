"""
storymix_v1 Stage B: ROC-only adaptation from the Stage A best checkpoint.

This is a warm-start stage: load model weights from Stage A, but reset optimizer
state and training counters for a clean ROC-focused adaptation run.
"""

out_dir = "out-storymix-v1-stageB"
init_from = "warmstart_path"
resume_ckpt_path = "out-storymix-v1-stageA/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "storymix-v1-stageB-roc-adapt"

dataset = "rocstories"
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

learning_rate = 1.2e-4
weight_decay = 7e-2
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1.2e-5
beta2 = 0.99
warmup_iters = 0

compile = False
seed = 2027
