"""
tiny_roc_curriculum_v2 Stage C: optional conservative continuation-weighted ROC polish.

Prepared as a backup file only. Do not run by default; only use it if Stage B
shows clear value and is already closing the gap to the current E4 bar.
"""

out_dir = "out-tiny-roc-curriculum-v2-stageC"
init_from = "warmstart_path"
resume_ckpt_path = "out-tiny-roc-curriculum-v2-stageB/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "tiny-roc-curriculum-v2-stageC-continuation"

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
loss_mode = "continuation_weighted"
prompt_weight = 0.8
continuation_weight = 1.15
ending_weight = 1.10
ending_tokens = 16

learning_rate = 5e-5
weight_decay = 7e-2
max_iters = 1000
lr_decay_iters = 1000
min_lr = 5e-6
beta2 = 0.99
warmup_iters = 0

compile = False
seed = 2027
