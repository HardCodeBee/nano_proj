# Task 2 Experiments

## Scope

This document is the single detailed record for Task 2.
It replaces the older plan-by-plan notes and keeps the information needed for later report writing in one place.

## Evaluation protocol

Task 2 now uses a fixed three-part evaluation setup:

1. `avg_loss / ppl` via the unmodified course `eval.py`
2. fixed-prompt generation using [task2_eval_prompts.txt](C:/Users/12442/Desktop/GitHub/nano_proj/prompts/task2_eval_prompts.txt)
3. a local OpenAI-compatible automatic judge proxy

For ROCStories, the daily protocol is:

- `train`: official train minus a held-out slice
- `val`: 5% slice of official train, used for daily comparison
- `locked_test`: official public test, used only for occasional shortlisted checks

Important comparability caveat:

- the historical `r19` checkpoint was trained before this split policy existed
- it saw the full official ROCStories train split
- so `r19` on the new `ROC val` is only a leakage sanity check, not a fair Task 2 baseline

## Historical Task 1 reference

- best documented artifact: `out-rocstories-remote-r19/`
- public-test result: `avg_loss = 3.216`, `ppl = 24.93`
- local public-test OpenAI-judge proxy: `2.0` was not achieved; the recorded proxy reference is `1.9`
- frozen reproduction recipe: [train_rocstories_r19_frozen.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_r19_frozen.py)

This historical Task 1 result remains valid and is not changed by Task 2 protocol updates.

## E1: TinyStories -> ROCStories curriculum

Configs:

- [train_tinystories_task2_e1_stage1.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_tinystories_task2_e1_stage1.py)
- [train_rocstories_task2_e1_stage2.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e1_stage2.py)

Idea:

- Stage 1 uses a clean TinyStories subset to teach cleaner short-range narrative transitions
- Stage 2 resumes on ROCStories to pull the model back to the exact course domain

Confirmed daily result:

- run: `e1-tinystories-to-rocstories`
- eval input: `ROC val`
- `avg_loss = 3.181`
- `ppl = 24.07`
- local automatic-judge proxy: `2.0`
- repetition failures: `1`
- truncation failures: `0`
- prompt drift failures: `0`

Takeaway:

- E1 was the first clearly useful Task 2 result
- it showed that narrative curriculum learning was technically feasible and directionally useful

## E2: story-aware ROCStories polish

Config:

- [train_rocstories_task2_e2_storyaware_polish.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e2_storyaware_polish.py)

Supporting code:

- [train.py](C:/Users/12442/Desktop/GitHub/nano_proj/train.py)
- [prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories/prepare.py)

Idea:

- keep the E1 checkpoint
- continue ROCStories training with a mixed story-aware sampler biased toward story openings
- use a short low-learning-rate polish stage

Confirmed daily result:

- run: `e2-storyaware-polish-openai`
- eval input: `ROC val`
- `avg_loss = 3.176`
- `ppl = 23.95`
- local automatic-judge proxy: `2.0`
- repetition failures: `2`
- truncation failures: `0`
- prompt drift failures: `0`

Takeaway:

- E2 gave a small but real token-level improvement over E1
- it became the current best daily `ROC val` checkpoint
- however, it did not create a clear judged story-quality jump

## E3: prompt-aligned synthetic distillation

Configs:

- [train_rocstories_synth_task2_e3_stage1.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_synth_task2_e3_stage1.py)
- [train_rocstories_task2_e3_stage2.py](C:/Users/12442/Desktop/GitHub/nano_proj/config/train_rocstories_task2_e3_stage2.py)

Supporting code:

- [generate_rocstories_synthetic.py](C:/Users/12442/Desktop/GitHub/nano_proj/scripts/generate_rocstories_synthetic.py)
- [task2_rocstyle_rewrite_prompt.txt](C:/Users/12442/Desktop/GitHub/nano_proj/prompts/task2_rocstyle_rewrite_prompt.txt)
- [prepare.py](C:/Users/12442/Desktop/GitHub/nano_proj/data/rocstories_synth/prepare.py)

Idea:

- generate short synthetic stories with a stronger model
- prioritize opening-sentence adherence, coherent continuation, and natural ending
- distill that data into the small checkpoint
- then return to ROCStories for a final polish

Confirmed daily result:

- run: `e3-synth-distill-openai`
- eval input: `ROC val`
- `avg_loss = 3.176`
- `ppl = 23.96`
- local automatic-judge proxy: `2.0`
- repetition failures: `0`
- truncation failures: `0`
- prompt drift failures: `0`

Important confirmation:

- this run used the corrected prompt-adherence generator
- this was verified from synthetic raw records containing `source_opening`

Takeaway:

- E3 looked slightly cleaner on simple failure heuristics
- but it did not beat E2 on `ppl`
- and it still did not lift the local judge out of the `2.0` band

## Main findings so far

The current evidence supports these conclusions:

1. Better data and better sampling can improve token-level fit a bit.
2. Those same changes have not yet produced a clear jump in prompt-conditioned story quality.
3. For this small model, improving `ppl` appears easier than improving judged story quality.

## Practical recommendation

- Keep E2 as the current best daily `ROC val` checkpoint.
- Treat E3 as an informative mixed result, not the new winner.
- If more exploration time remains, the next experiment should change the training target more directly around prompt-conditioned continuation instead of only changing the source corpus.

## Files that matter most now

- [task2_current_status.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/task2_current_status.md)
- [task2_experiments.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/task2_experiments.md)
- [results.csv](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/results.csv)
