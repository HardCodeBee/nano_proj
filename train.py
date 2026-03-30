"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'resume_path' or 'gpt2*'
resume_ckpt_path = '' # used only when init_from='resume_path'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters
sampling_mode = 'random' # random, story_start, opening_biased, mixed
story_sampling_prob = 0.7 # for mixed mode: probability of drawing story-aware windows
opening_window_tokens = 24 # max offset from a story start for opening-biased sampling
loss_mode = 'standard' # standard, continuation_weighted
prompt_weight = 0.5 # relative weight for tokens inside the opening sentence
continuation_weight = 2.0 # relative weight for continuation tokens after the opening sentence
ending_weight = 1.25 # extra multiplier for the final ending_tokens continuation targets
ending_tokens = 16 # number of continuation targets near story end to boost
mask_after_story_end = False # ignore targets that spill past the current story's first EOT
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
seed = 1337 # make seed configurable so experiments can sweep initialization cleanly
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
if master_process:
    print(f"train sampling mode: {sampling_mode}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
story_metadata_cache = {}
story_sampling_warning_printed = False
EOT_TOKEN_ID = 50256


def load_story_metadata(split, require_first_sentence=False):
    global story_sampling_warning_printed
    cache_key = (data_dir, split, require_first_sentence)
    if cache_key in story_metadata_cache:
        return story_metadata_cache[cache_key]

    starts_path = os.path.join(data_dir, f'{split}_story_starts.npy')
    lengths_path = os.path.join(data_dir, f'{split}_story_lengths.npy')
    first_sentence_lengths_path = os.path.join(data_dir, f'{split}_first_sentence_lengths.npy')
    if not (os.path.exists(starts_path) and os.path.exists(lengths_path)):
        if sampling_mode != 'random' and split == 'train' and master_process and not story_sampling_warning_printed:
            print(
                f"story-aware sampling requested, but metadata is missing for split '{split}'. "
                "Falling back to random stream sampling."
            )
            story_sampling_warning_printed = True
        story_metadata_cache[cache_key] = None
        return None

    starts = np.load(starts_path).astype(np.int64, copy=False)
    lengths = np.load(lengths_path).astype(np.int64, copy=False)
    if require_first_sentence:
        if not os.path.exists(first_sentence_lengths_path):
            raise FileNotFoundError(
                f"Missing continuation metadata for split '{split}': {first_sentence_lengths_path}. "
                "Re-run the dataset prepare script before using continuation-weighted loss."
            )
        first_sentence_lengths = np.load(first_sentence_lengths_path).astype(np.int64, copy=False)
    elif os.path.exists(first_sentence_lengths_path):
        first_sentence_lengths = np.load(first_sentence_lengths_path).astype(np.int64, copy=False)
    else:
        first_sentence_lengths = None
    story_metadata_cache[cache_key] = {
        'starts': starts,
        'lengths': lengths,
        'first_sentence_lengths': first_sentence_lengths,
    }
    return story_metadata_cache[cache_key]


def sample_random_indices(num_samples, data_length):
    return np.random.randint(0, data_length - block_size, size=num_samples).astype(np.int64)


def sample_story_indices(num_samples, data_length, split, mode):
    metadata = load_story_metadata(split)
    if metadata is None:
        return sample_random_indices(num_samples, data_length)

    valid_limit = data_length - block_size
    valid_mask = metadata['starts'] < valid_limit
    starts = metadata['starts'][valid_mask]
    lengths = metadata['lengths'][valid_mask]
    if len(starts) == 0:
        return sample_random_indices(num_samples, data_length)

    chosen = np.random.randint(0, len(starts), size=num_samples).astype(np.int64)
    chosen_starts = starts[chosen]
    if mode == 'story_start':
        return chosen_starts

    if mode == 'opening_biased':
        max_offsets = np.minimum(np.maximum(lengths[chosen] - 2, 0), opening_window_tokens)
        offsets = np.array(
            [np.random.randint(0, offset + 1) if offset > 0 else 0 for offset in max_offsets],
            dtype=np.int64,
        )
        return chosen_starts + offsets

    return sample_random_indices(num_samples, data_length)


def choose_start_indices(split, data_length):
    if split != 'train' or sampling_mode == 'random':
        return sample_random_indices(batch_size, data_length)

    if sampling_mode in ('story_start', 'opening_biased'):
        return sample_story_indices(batch_size, data_length, split, sampling_mode)

    if sampling_mode == 'mixed':
        use_story = np.random.rand(batch_size) < story_sampling_prob
        ix = sample_random_indices(batch_size, data_length)
        story_count = int(use_story.sum())
        if story_count > 0:
            ix[use_story] = sample_story_indices(story_count, data_length, split, 'opening_biased')
        return ix

    raise ValueError(f"Unknown sampling_mode: {sampling_mode}")


def prepare_targets_and_weights(start_indices, targets, split):
    apply_story_end_mask = mask_after_story_end or loss_mode == 'continuation_weighted'
    if not apply_story_end_mask:
        if loss_mode != 'standard':
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        return targets, None

    metadata = load_story_metadata(split, require_first_sentence=(loss_mode == 'continuation_weighted'))
    if metadata is None:
        raise ValueError(
            f"Story-bounded loss requested for split '{split}', but story metadata is unavailable in {data_dir}."
        )

    starts = metadata['starts']
    lengths = metadata['lengths']
    first_sentence_lengths = metadata['first_sentence_lengths']
    if len(starts) == 0:
        return targets, None

    masked_targets = targets.clone()
    weights = None
    if loss_mode == 'continuation_weighted':
        weights = torch.zeros_like(masked_targets, dtype=torch.float32)
    elif loss_mode != 'standard':
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    story_ids = np.searchsorted(starts, start_indices, side='right') - 1
    story_ids = np.clip(story_ids, 0, len(starts) - 1)
    seq_len = masked_targets.size(1)

    for row, story_id in enumerate(story_ids.tolist()):
        start_idx = int(start_indices[row])
        story_start = int(starts[story_id])
        story_end = story_start + int(lengths[story_id]) - 1
        valid_len = max(0, min(seq_len, story_end - start_idx))

        if valid_len < seq_len:
            masked_targets[row, valid_len:] = -1

        if weights is None:
            continue

        prompt_end = story_start + int(first_sentence_lengths[story_id]) - 1
        prompt_valid_len = max(0, min(valid_len, prompt_end - start_idx))

        if prompt_valid_len > 0:
            weights[row, :prompt_valid_len] = prompt_weight
        if valid_len > prompt_valid_len:
            weights[row, prompt_valid_len:valid_len] = continuation_weight
            if ending_weight != 1.0 and ending_tokens > 0:
                tail_start = max(prompt_valid_len, valid_len - ending_tokens)
                weights[row, tail_start:valid_len] *= ending_weight

    return masked_targets, weights


def compute_loss_for_batch(model, inputs, targets, start_indices, split):
    effective_targets, loss_weights = prepare_targets_and_weights(start_indices, targets, split)
    logits, standard_loss = model(inputs, effective_targets)
    if loss_weights is None:
        return logits, standard_loss

    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        effective_targets.view(-1),
        ignore_index=-1,
        reduction='none',
    ).view_as(effective_targets)
    loss_weights = loss_weights.to(per_token_loss.dtype)
    weighted_loss = (per_token_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
    return logits, weighted_loss


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = choose_start_indices(split, len(data))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from in ('resume', 'resume_path'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') if init_from == 'resume' else resume_ckpt_path
    if not ckpt_path:
        raise ValueError("resume_ckpt_path must be set when init_from='resume_path'")
    print(f"Resuming training from {ckpt_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from in ('resume', 'resume_path'):
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, ix = get_batch(split)
            with ctx:
                logits, loss = compute_loss_for_batch(model, X, Y, ix, split)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, ix = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = compute_loss_for_batch(model, X, Y, ix, 'train')
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, ix = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
