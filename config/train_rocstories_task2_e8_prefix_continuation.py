"""
Task 2 E8 pilot: story-level opening -> continuation batching with a small auxiliary LM mix.
"""

out_dir = "out-task2-e8-prefix-continuation"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e4-posteot-mask/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e8-prefix-continuation"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "full_story"
loss_mode = "prefix_to_continuation"
mask_after_story_end = True
aux_lm_prob = 0.2
aux_sampling_mode = "mixed"
aux_loss_mode = "standard"
aux_mask_after_story_end = True
story_sampling_prob = 0.7
opening_window_tokens = 24

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 5e-5
weight_decay = 7e-2
max_iters = 16000
lr_decay_iters = 16000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 50

compile = False
seed = 2027
