"""
Task 2 E7: warm-start the best real-data checkpoint into a near-capacity 7-layer model with a longer context window.
"""

out_dir = "out-task2-e7-warmstart-depth7-bs128"
init_from = "warmstart_path"
resume_ckpt_path = "out-task2-e4-posteot-mask/ckpt.pt"
warmstart_copy_last_block = True

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e7-warmstart-depth7-bs128"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 56
sampling_mode = "mixed"
story_sampling_prob = 0.6
opening_window_tokens = 32
mask_after_story_end = True
loss_mode = "standard"

block_size = 128
n_layer = 7
n_head = 6
n_embd = 384
dropout = 0.12

learning_rate = 5e-5
weight_decay = 7e-2
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 200

compile = False
seed = 2027
