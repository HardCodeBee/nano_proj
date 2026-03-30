"""
Task 2 E4 Step 1: isolate post-EOT spillover by masking targets after the current story ends.
"""

out_dir = "out-task2-e4-posteot-mask"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e2-storyaware-polish/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e4-posteot-mask"

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
dropout = 0.14

learning_rate = 8e-5
weight_decay = 7e-2
max_iters = 14500
lr_decay_iters = 14500
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 50

compile = False
seed = 2027
