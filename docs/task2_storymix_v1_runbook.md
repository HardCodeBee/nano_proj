# Task 2 storymix_v1 runbook

This is the new default mainline for Task 2 work.
Older E4/E5/E6/E7/E8 branches stay in the repo as historical comparisons, not as the active path.

## Stages

- Stage A: scratch pretraining on `storymix_v1`
  Config: `config/train_storymix_v1_stageA_scratch.py`
  Goal: build a narrative-only base model from random initialization with ROC train + filtered TinyStories.

- Stage B: ROC-only adaptation
  Config: `config/train_storymix_v1_stageB_roc_adapt.py`
  Goal: warm-start from the Stage A best checkpoint, reset optimizer state, and retarget to ROCStories only.

- Stage C: conservative continuation polish
  Config: `config/train_storymix_v1_stageC_continuation.py`
  Goal: warm-start from the Stage B best checkpoint and apply a mild continuation-weighted polish on ROCStories.

## Recommended order

1. Prepare `data/rocstories` if it is missing.
2. Run `python data/storymix_v1/prepare.py`.
3. Train Stage A from scratch.
4. Train Stage B from the Stage A best checkpoint.
5. Train Stage C from the Stage B best checkpoint.
6. Run `scripts/run_decode_sweep.py` on shortlisted Stage B and Stage C checkpoints.
7. Run `scripts/analyze_task2_samples.py` on the sample JSONL files you want to compare side by side.

## Stop/go checks

- After dataset prep:
  Check `data/storymix_v1/filter_report.json` and `data/storymix_v1/dataset_stats.json`.
  If TinyStories selection is far below target or the kept-length distribution drifts badly from ROC, fix the filters before training.

- After Stage A:
  Continue only if training is stable and samples look like short multi-sentence narratives rather than TinyStories-style fragments.

- After Stage B:
  Use Stage B as the default shortlist checkpoint unless Stage C clearly improves continuation quality without a material ROC val PPL regression.

- After Stage C:
  Treat Stage C as a conservative polish, not an automatic winner.
  If samples become more repetitive, less prompt-faithful, or PPL degrades too much, fall back to Stage B.

## Submission choice

- Compare Stage B vs Stage C on the same prompt set and decode sweep.
- Prefer Stage C only if it wins on sample quality while keeping ROC val fit close enough to Stage B.
- If the tradeoff is ambiguous, default to Stage B.
