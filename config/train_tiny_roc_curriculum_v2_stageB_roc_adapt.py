"""
tiny_roc_curriculum_v2 Stage B: ROC-only main adaptation warm-started from Stage A.

This stage reuses Stage A model weights but resets optimizer state, iter_num,
and best_val_loss via train.py's existing warmstart_path logic.
"""

out_dir = "out-tiny-roc-curriculum-v2-stageB"
init_from = "warmstart_path"
resume_ckpt_path = "out-tiny-roc-curriculum-v2-stageA/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "tiny-roc-curriculum-v2-stageB-roc-adapt"

dataset = "rocstories"
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

learning_rate = 2.0e-4
weight_decay = 7e-2
max_iters = 8000
lr_decay_iters = 8000
min_lr = 2e-5
beta2 = 0.99
warmup_iters = 100

compile = False
seed = 2027
