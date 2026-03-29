# Task 2 Current Status

## Snapshot

This is the current practical conclusion after running the main Task 2 branches.

For the detailed experiment record, see
[task2_experiments.md](C:/Users/12442/Desktop/GitHub/nano_proj/out-task2/task2_experiments.md).

## What is stable

- Task 1 historical best remains valid:
  - `r19` public-test `avg_loss = 3.216`
  - `r19` public-test `ppl = 24.93`
- Task 2 daily comparison now uses the new `ROC val` protocol.
- Old `r19` on the new `ROC val` is not a fair baseline and should not be used as the main comparison line.

## Main experimental branches so far

### E1: TinyStories -> ROCStories curriculum

- `avg_loss = 3.181`
- `ppl = 24.07`
- local automatic-judge proxy: `2.0`

Conclusion:

- this was the first genuinely useful Task 2 result
- it proved the curriculum direction was viable

### E2: story-aware ROCStories polish

- `avg_loss = 3.176`
- `ppl = 23.95`
- local automatic-judge proxy: `2.0`

Conclusion:

- this gave a small but real token-level improvement over E1
- it became the current best daily `ROC val` checkpoint
- however, story-quality gains were modest

### E3: prompt-aligned synthetic distillation + ROCStories polish

- corrected prompt-adherence synthetic data was actually used
- `avg_loss = 3.176`
- `ppl = 23.96`
- local automatic-judge proxy: `2.0`
- repetition / truncation / drift failure counts looked cleaner than E2

Conclusion:

- E3 did not produce the hoped-for story-quality jump
- it looked slightly cleaner on some failure heuristics
- but it did not beat E2 on `ppl`, and did not raise judge scores above `2`

## Current best interpretation

The project has already found two things that are true at the same time:

1. Data and sampling changes can improve token-level fit a bit.
2. Those same changes have not yet produced a clear jump in prompt-conditioned story quality.

So the current bottleneck is no longer simply "find better data" or "find slightly better sampling".
The deeper issue is that the model still struggles to consistently turn an opening sentence into a coherent short story with a natural ending.

## What to say in the report right now

- Task 2 exploration was real and evidence-based.
- The TinyStories curriculum was the first successful direction.
- Story-aware polish gave a modest incremental gain.
- Prompt-aligned synthetic distillation was more aggressive and more novel, but still did not produce a decisive quality breakthrough.
- Therefore, the main finding is not just "what worked" but also that improving `ppl` is easier than improving judged story quality for this small model.

## Current recommendation

- Keep E2 as the current best daily `ROC val` checkpoint.
- Treat E3 as an informative negative-or-mixed result rather than the final answer.
- If more Task 2 time remains, the next experiment should change the training target more directly around prompt-conditioned continuation rather than only changing the source data.
