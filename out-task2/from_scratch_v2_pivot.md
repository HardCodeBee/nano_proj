# From-scratch v2 pivot

## Why `storymix_v1` is currently judged a failure

- `storymix_v1` Stage A: `ppl 27.19`, `judge 2.0`
- `storymix_v1` Stage B: `ppl 27.70`, `judge 2.0`
- Stage B regressed against Stage A instead of improving it.
- Both runs still sit clearly below the current scored bar `e4-posteot-mask-openai`.
- Therefore `storymix_v1 Stage C` should not continue on the active path, because it depends on a failed Stage B.

## Why v2 pivots to short curriculum pretraining

- We keep the from-scratch and Task 1-compliant backbone.
- We stop using broad mixed-data pretraining as the mainline.
- We reduce external narrative data to a short filtered TinyStories curriculum.
- We move most target-domain learning back into ROC-only adaptation.

## What each v2 config does

- `config/train_tiny_roc_curriculum_v2_stageA_tiny_short.py`
  Short filtered TinyStories pretrain from scratch. This is only a light narrative prior.
- `config/train_tiny_roc_curriculum_v2_stageB_roc_adapt.py`
  ROC-only main adaptation warm-started from Stage A. This is the actual main training stage.
- `config/train_tiny_roc_curriculum_v2_stageC_continuation.py`
  Optional conservative continuation-weighted ROC polish. This is a backup file, not the default next step.

## Recommended remote order

1. Prepare `data/tinystories_rocstyle_v2`.
2. Prepare `data/rocstories` if needed on the remote machine.
3. Run Stage A.
4. Run Stage B.
5. Only if Stage B passes the gate below, consider Stage C.

## Gate rule

- Default order is always Stage A then Stage B.
- Only consider Stage C if Stage B clearly beats `storymix_v1 Stage A/B` and starts closing the gap to the current E4 bar.
- Otherwise Stage C does not run.
