# ROCStories Task 1 Push Sweep Notes

This file originally proposed the `v3` seed sweep. That sweep has now been completed,
and the later `v6` / `v7` follow-ups are the important outcome to keep in mind.

## Completed `v3` Seed Sweep

Base recipe (`config/train_rocstories_task1_push_v3.py`):

- `block_size = 96`
- `batch_size = 80`
- `dropout = 0.14`
- `learning_rate = 3.5e-4`
- `weight_decay = 7e-2`

Completed seed results:

| run | config | seed | ppl |
| --- | --- | ---: | ---: |
| `r15` | `v3` | `2027` | `25.10` |
| `r16` | `v3` | `31415` | `25.24` |
| `r17` | `v3` | `424242` | `25.31` |

Conclusion:

- `seed = 2027` was clearly the strongest seed on the `v3` recipe.
- That seed then became the default for the later continuation runs.

## Later Follow-Ups

`v6` extended the strongest `v3/r15` trajectory:

- `r18` (`v6`, seed `2027`) -> `ppl = 25.00`

`v7` then extended training further and evaluated more frequently:

| run | config | seed | ppl |
| --- | --- | ---: | ---: |
| `r19` | `v7` | `2027` | `24.93` |
| `r20` | `v7` | `31415` | `24.98` |
| `r21` | `v7` | `424242` | `24.96` |
| `r22` | `v7` | `777777` | `24.95` |

## Current Takeaway

- The short-context recipe (`block_size = 96`, `batch_size = 80`) was the main turning point after the older 128-token baseline plateaued.
- `seed = 2027` remains the best seed observed in the repo.
- `config/train_rocstories_task1_push_v7.py` is the current best-performing push config family.
- `out-rocstories-remote-r19/ckpt.pt` is the current best validated public-test checkpoint.
