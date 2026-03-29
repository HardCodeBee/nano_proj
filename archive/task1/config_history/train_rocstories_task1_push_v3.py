"""
Third ROCStories Task 1 push after the 25.34-25.35 plateau.

Key idea:
- ROCStories are short (public-val max length is 91 GPT-2 tokens in dataset_stats)
- shrink block_size so the model focuses on the story lengths we actually evaluate
- raise batch_size to keep tokens/iteration close to prior successful runs
"""

out_dir = "out-rocstories-task1-push-v3"
eval_interval = 100
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push-v3"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 80
block_size = 96

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 3.5e-4
weight_decay = 7e-2
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 1337
