"""
Task 2 E1 Stage 1: narrative curriculum pretraining on TinyStories.

Design:
- Keep the exact baby-GPT architecture and tokenizer pipeline from the ROCStories baseline
- Keep block_size=96 so Stage 2 can resume training without checkpoint surgery
- Spend only part of the total budget on TinyStories, then return to ROCStories

Planned schedule:
- Stage 1: run this config from scratch to 4k steps on TinyStories
- Stage 2: resume the same out_dir with config/train_rocstories_task2_e1_stage2.py
"""

out_dir = "out-task2-e1-tinystories-to-rocstories"
eval_interval = 50
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "rocstories"
wandb_run_name = "task2-e1-stage1-tinystories"

dataset = "tinystories"
gradient_accumulation_steps = 1
batch_size = 80

# Keep the checkpoint shape identical to the ROCStories baseline so resume works cleanly.
block_size = 96
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.14

learning_rate = 3.5e-4
weight_decay = 7e-2
max_iters = 4000
lr_decay_iters = 12000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 500

compile = False
seed = 2027
