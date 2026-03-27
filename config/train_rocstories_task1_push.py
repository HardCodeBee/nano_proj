"""
Low-risk Task 1 follow-up from the 25.57 PPL ROCStories baseline.

Intent:
- keep the official baby-GPT architecture and unchanged tokenizer/data pipeline
- add a bit more regularization for this small corpus
- evaluate twice as often so the best checkpoint is less likely to fall between evals
"""

out_dir = "out-rocstories-task1-push"
eval_interval = 100
eval_iters = 100
log_interval = 10

# Save only the best checkpoint seen on validation.
always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "baby-gpt-rocstories-task1-push"

dataset = "rocstories"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.12

learning_rate = 4e-4
weight_decay = 6e-2
max_iters = 9000
lr_decay_iters = 9000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 400

compile = False
seed = 1337
