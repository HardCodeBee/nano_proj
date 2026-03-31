"""
Task 2 E5 ROC-native Stage 2: short real-ROC recovery that keeps post-EOT masking on.
"""

out_dir = "out-task2-e5-roc-native-masked-recovery"
init_from = "resume_path"
resume_ckpt_path = "out-task2-e5-roc-native-synth-continuation-weighted/ckpt.pt"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e5-roc-native-masked-recovery"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.6
opening_window_tokens = 24
loss_mode = "standard"
mask_after_story_end = True

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 2.5e-5
weight_decay = 7e-2
max_iters = 16000
lr_decay_iters = 16000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 25

compile = False
seed = 2027
