# ROCStories Task 1 V3 Seed Sweep

Current best configuration is `config/train_rocstories_task1_push_v3.py`:

- `block_size = 96`
- `batch_size = 80`
- `dropout = 0.14`
- `learning_rate = 3.5e-4`
- `weight_decay = 7e-2`

After `r12` reached `25.16 PPL`, nearby hyperparameter changes (`v4`, `v5`) regressed.
The next highest-probability move is to keep the `v3` recipe fixed and sweep seeds.

Suggested runs:

```bash
python train.py config/train_rocstories_task1_push_v3.py --out_dir=out-rocstories-remote-r15 --device=cuda --dtype=bfloat16 --seed=2027 2>&1 | tee out-rocstories-remote-r15/train.log
python eval.py --init_from=resume --out_dir=out-rocstories-remote-r15 --input_file=data/rocstories/test_full.txt --print_first_n=0 2>&1 | tee out-rocstories-remote-r15/eval_test_full.log
```

```bash
python train.py config/train_rocstories_task1_push_v3.py --out_dir=out-rocstories-remote-r16 --device=cuda --dtype=bfloat16 --seed=31415 2>&1 | tee out-rocstories-remote-r16/train.log
python eval.py --init_from=resume --out_dir=out-rocstories-remote-r16 --input_file=data/rocstories/test_full.txt --print_first_n=0 2>&1 | tee out-rocstories-remote-r16/eval_test_full.log
```

```bash
python train.py config/train_rocstories_task1_push_v3.py --out_dir=out-rocstories-remote-r17 --device=cuda --dtype=bfloat16 --seed=424242 2>&1 | tee out-rocstories-remote-r17/train.log
python eval.py --init_from=resume --out_dir=out-rocstories-remote-r17 --input_file=data/rocstories/test_full.txt --print_first_n=0 2>&1 | tee out-rocstories-remote-r17/eval_test_full.log
```
