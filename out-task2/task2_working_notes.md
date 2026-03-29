# Task 2 Working Notes

This file is a report-ready working log for Task 2. It records what has already been tried,
why the direction fits the assignment, what evidence we currently have, and which caveats
must be stated clearly in the final report.

## 1. Task 2 goal and assignment fit

Task 2 is not a free-for-all exploration detached from story generation. The assignment
clarification says the exploration should still support story generation, and Task 3 must
submit the best improved story checkpoint rather than the plain Task 1 baseline.

That makes the current direction well aligned with the course requirements:

- final model stays within the `<= 32M` constraint
- default `eval.py` remains unchanged
- default text preprocessing compatibility is preserved
- exploration is still aimed at improving story generation quality

## 2. Chosen exploration direction

### Main idea

Use a two-stage narrative curriculum:

1. Stage 1: train on a clean TinyStories subset
2. Stage 2: resume from that checkpoint and adapt back to ROCStories

This is more than simply swapping the dataset. The hypothesis is:

- TinyStories can teach cleaner local narrative transitions and endings
- ROCStories Stage 2 can pull the model back to the exact target domain
- together they may improve both story coherence and final story quality

### Why TinyStories was chosen

- it remains a story-generation dataset, so it stays on-task
- it is cleaner and more self-contained than longer/noisier web-story corpora
- it matches the current small-model, short-context setup better than very long story sources

## 3. Data and protocol decisions

### ROCStories protocol

Task 2 now uses a three-layer ROCStories split:

- `train`: official train minus a held-out slice
- `val`: 5% slice of official train, used for daily comparison
- `locked_test`: official ROCStories test split, used only for shortlisted checks

This protocol is implemented in:
[prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories/prepare.py)

### TinyStories protocol

TinyStories is auxiliary curriculum data only:

- `train`: subset for Stage 1 training
- `val`: small subset for sanity checking Stage 1
- no separate TinyStories test split is needed

This protocol is implemented in:
[prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/tinystories/prepare.py)

### Fixed evaluation setup

All Task 2 runs should use the same three evaluation axes:

1. `avg_loss / ppl` via the unmodified course `eval.py`
2. fixed-prompt story generation
3. automatic judge scoring through the OpenAI-compatible local scoring path

This runner is implemented in:
[task2_generate_and_score.py](C:/Users/12442/Desktop/GitHub/nano_proj/scripts/task2_generate_and_score.py)

## 4. Completed experiments

### Historical Task 1 reference

Frozen historical baseline:

- checkpoint directory: `out-rocstories-remote-r19`
- public-test result: `avg_loss = 3.216`, `ppl = 24.93`
- local OpenAI-judge proxy on the same public-test protocol: `mean score = 1.9`

Important:

- this remains the canonical Task 1 best result
- this result is still valid and should not be rewritten by Task 2 protocol changes

### E1: TinyStories -> ROCStories curriculum

Configs:

- Stage 1: [train_tinystories_task2_e1_stage1.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_tinystories_task2_e1_stage1.py)
- Stage 2: [train_rocstories_task2_e1_stage2.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e1_stage2.py)

Current confirmed daily result under the new protocol:

- run name: `e1-tinystories-to-rocstories`
- eval input: `ROC val`
- `avg_loss = 3.181`
- `ppl = 24.07`
- repetition failures on fixed prompts: `1`
- truncation failures: `0`
- prompt drift failures: `0`

Evidence files:

- [results.csv](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/results.csv)
- [e1-tinystories-to-rocstories_eval.txt](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2-1/eval_logs/e1-tinystories-to-rocstories_eval.txt)
- [e1-tinystories-to-rocstories.jsonl](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2-1/samples/e1-tinystories-to-rocstories.jsonl)

Interpretation:

- this is a valid Task 2 signal under the new split protocol
- the model is clearly no longer a TinyStories-only checkpoint
- the generated stories still show semantic jumps and weak realism, so the direction is promising but not yet final

### E2: story-aware ROCStories polish

Configs and code path:

- sampler logic: [train.py](C:/Users/12442/Desktop/GitHub/nano_proj/train.py)
- ROCStories metadata export: [prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories/prepare.py)
- polish config: [train_rocstories_task2_e2_storyaware_polish.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e2_storyaware_polish.py)

Current confirmed result under the new protocol:

- run name: `e2-storyaware-polish-openai`
- eval input: `ROC val`
- `avg_loss = 3.176`
- `ppl = 23.95`
- OpenAI-judge proxy mean score: `2.0`
- repetition failures: `2`
- truncation failures: `0`
- prompt drift failures: `0`

Interpretation:

- E2 is a small but real token-level improvement over E1
- the automatic judge score did not improve
- some local stories are slightly cleaner, but overall story quality is still mostly in the same band
- E2 should be treated as the current best daily ROC-val checkpoint, but not yet as a decisive final model

## 5. Important comparability caveat

The old `r19` checkpoint was trained before the new Task 2 split protocol existed.
It saw the full official ROCStories train split.

Therefore:

- `r19` evaluated on the new `ROC val` is not a fair apples-to-apples baseline
- the observed `avg_loss = 2.495`, `ppl = 12.13` on `ROC val` is a leakage sanity check, not a valid Task 2 comparison

For the report:

- keep `r19 public test = 24.93` as the historical Task 1 reference
- compare new Task 2 daily `ROC val` numbers only among models trained under the same split policy

## 6. What worked and what did not

### Worked

- the TinyStories -> ROCStories curriculum is technically feasible in the current codebase
- Stage 2 resume into ROCStories works after forcing checkpoint saving
- the fixed evaluation runner gives consistent metrics and sample outputs
- story-aware ROCStories polish gave a small but positive ROC-val improvement over E1

### Did not work / issues encountered

- evaluating the Stage 1 checkpoint directly on ROCStories gave very poor transfer
- the first Stage 2 run failed to overwrite the Stage 1 checkpoint because `always_save_checkpoint` was false
- a locally copied E1 `ckpt.pt` became unreadable, so large checkpoint transfer should not be relied on for daily work
- the historical `r19` checkpoint cannot be fairly compared on the new `ROC val`
- the first story-aware polish run improved `ppl` only slightly and did not improve the automatic judge score

## 7. Failure modes observed so far

From the current fixed-prompt samples:

- occasional repetitive phrasing
- semantic drift inside the middle of the story
- endings are less abrupt than the TinyStories-only model, but still not consistently natural
- story-aware polish slightly cleaned a few prompts, but many samples remain bland or loosely connected

These failure modes still match the rubric concerns:

- coherence
- repetition
- following the opening sentence
- natural conclusion

## 8. Next high-value experiment

The next aggressive branch is:

`E3 = synthetic ROC-style distillation + final ROCStories polish`

Why:

- E1 and E2 improved token-level fit, but did not create a clear jump in judged story quality
- a stronger model can generate short synthetic stories that stay much closer to the opening prompt we want the small model to follow
- this keeps the task fully focused on story generation while being more novel than another small sampler tweak
- it is more likely to produce a story-quality jump than RLHF/DPO under the current time and compute constraints

Implementation files:

- [generate_rocstories_synthetic.py](C:/Users/12442/Desktop/GitHub/nano_proj/scripts/generate_rocstories_synthetic.py)
- [task2_rocstyle_rewrite_prompt.txt](C:/Users/12442/Desktop/GitHub/nano_proj/prompts/task2_rocstyle_rewrite_prompt.txt)
- [prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories_synth/prepare.py)
- [train_rocstories_synth_task2_e3_stage1.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_synth_task2_e3_stage1.py)
- [train_rocstories_task2_e3_stage2.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e3_stage2.py)
- [e3_synthetic_distillation_plan.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/e3_synthetic_distillation_plan.md)

## 9. Report writing checklist

The final report should be able to pull directly from the following items:

- motivation for TinyStories curriculum
- ROCStories split policy and evaluation protocol
- E1 configuration summary
- E2 story-aware polish summary
- E3 synthetic-distillation plan and rationale
- historical `r19` Task 1 reference
- current E1 and E2 daily ROC-val results
- short error analysis from fixed-prompt outputs
- explicit caveat that old `r19` on new `ROC val` is not a fair comparison

## 10. Files to cite when writing the report

- [Mini-Project 1 NanoGPT.txt](C:/Users/12442/Desktop/GitHub/nano_proj/instruction/Mini-Project%201%20NanoGPT.txt)
- [ClarificationsonTasks123.md](C:/Users/12442/Desktop/GitHub/nano_proj/instruction/ClarificationsonTasks123.md)
- [r19_baseline.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/r19_baseline.md)
- [task2_eval_protocol.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/task2_eval_protocol.md)
- [tinystories_curriculum_plan.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/tinystories_curriculum_plan.md)
- [results.csv](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/results.csv)
