"""
Task 2 E3 Stage 2: final ROCStories polish after synthetic ROC-style distillation.
"""

out_dir = "out-task2-e3-synth-distill"
init_from = "resume"

eval_interval = 25
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e3-stage2-rocstories-polish"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
sampling_mode = "mixed"
story_sampling_prob = 0.8
opening_window_tokens = 12

block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 8.0e-5
weight_decay = 7e-2
max_iters = 16000
lr_decay_iters = 16000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 100

compile = False
seed = 2027
