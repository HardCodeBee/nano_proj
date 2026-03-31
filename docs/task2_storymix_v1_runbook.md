# Task 2 storymix_v1 runbook

`storymix_v1` is now a historical Task 2 attempt, not the active default mainline.
It was the first from-scratch mainline candidate for this phase; older E4/E5/E6/E7/E8
branches also remain in the repo as historical comparisons.

## Stages

- Stage A: scratch pretraining on `storymix_v1`
  Config: `config/train_storymix_v1_stageA_scratch.py`
  Goal: build a narrative-only base model from random initialization with ROC train + filtered TinyStories.

- Stage B: ROC-only adaptation
  Config: `config/train_storymix_v1_stageB_roc_adapt.py`
  Goal: warm-start from the Stage A best checkpoint, reset optimizer state, and retarget to ROCStories only.

- Stage C: conservative continuation polish
  Config: `config/train_storymix_v1_stageC_continuation.py`
  Goal: archived optional polish only, not the current next step.

## Historical order

1. Prepare `data/rocstories` if it is missing.
2. Run `python data/storymix_v1/prepare.py`.
3. Train Stage A from scratch.
4. Train Stage B from the Stage A best checkpoint.

Stop there for the current mainline decision. Do not keep advancing to Stage C
unless you are explicitly reproducing the archived ablation for record-keeping.

## Stop/go checks

- After dataset prep:
  Check `data/storymix_v1/filter_report.json` and `data/storymix_v1/dataset_stats.json`.
  If TinyStories selection is far below target or the kept-length distribution drifts badly from ROC, fix the filters before training.

- After Stage A:
  Continue only if training is stable and samples look like short multi-sentence narratives rather than TinyStories-style fragments.

- After Stage B:
  Treat the branch as failed for the active mainline unless new evidence says otherwise.
  The current recommendation is to pivot to `tiny_roc_curriculum_v2`, not to keep stacking fixes on top of Stage B.

- After Stage C:
  Archived only. Do not treat it as the next default experiment from the current repo state.

## Current pilot outcome

- Remote console result for Stage A on 2026-03-31: `3.303 / 27.19 / 2.0`.
- Official scored Stage B result: `3.322 / 27.70 / 2.0`.
- Official scored Stage C result: `3.390 / 29.66 / 2.2`.

Interpretation:

- Stage A learned a usable five-sentence story shell, but Stage B did not improve on it.
- Stage C recovered a small quality signal, but with too much `ppl` regression.
- The branch remains useful as a historical comparison, not the current winner.
- `e4-posteot-mask-openai` remains the practical Task 2 anchor until a different from-scratch route produces a materially better tradeoff.
