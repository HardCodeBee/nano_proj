# Task 2 Working Notes

## Purpose

This file keeps short narrative notes that are useful for later report writing.
It should stay lighter than `task2_experiments.md` and focus on interpretation rather than exhaustive inventory.

## Current report-ready storyline

- Task 1 established a valid ROCStories baseline, but the project goal moved to improving story generation for Task 2 / Task 3.
- E1 showed that a TinyStories -> ROCStories curriculum can help: it was the first clearly useful Task 2 result.
- E2 added story-aware ROCStories polish and became the current best daily `ROC val` checkpoint, but the gain was modest.
- E3 tested a more aggressive prompt-aligned synthetic-distillation branch. It was novel and correctly targeted opening adherence, but it still did not produce a decisive judged-quality jump.

## What is safe to claim

- The exploration was real, multi-stage, and evidence-based.
- Better data and better sampling improved token-level fit.
- For this small model, improving `ppl` turned out to be easier than improving judged story quality.
- The strongest practical checkpoint so far is E2, not because it is dramatically better, but because it is the cleanest small improvement that actually held up.

## What should be stated carefully

- Daily Task 2 comparisons should be made only among checkpoints trained and evaluated under the new held-out `ROC val` protocol; `locked_test` is for occasional shortlist checks.
- Old `r19` remains the historical Task 1 reference, but it is not a fair baseline on the new `ROC val`.
- E3 should not be oversold. It was a meaningful aggressive experiment, but it did not clearly beat E2.
- The local automatic judge is only a proxy for the final evaluation; it is useful, but not the official private-test scorer.

## Failure pattern summary

- The model can often stay on the general topic of the opening sentence.
- The harder part is sustaining a coherent event chain through the middle of the story.
- Natural endings remain inconsistent.
- Many samples are still bland even when they are not obviously broken.

## Best current interpretation

The main bottleneck is no longer simply dataset choice.
The deeper problem is that the model still struggles with prompt-conditioned continuation: turning an opening sentence into a compact, coherent, naturally ending short story.

## If writing the report now

Main Task 2 arc:

1. Start from a frozen historical ROCStories reference.
2. Test curriculum learning with TinyStories.
3. Add story-aware polish.
4. Try a more aggressive prompt-aligned synthetic-distillation branch.
5. Conclude that token-level improvements were easier to obtain than judged story-quality improvements.

## Next-step note

If experimentation continues, the next useful branch should change the training target more directly around opening-to-story continuation rather than only changing the source corpus again.
This round, that direction has been translated into code scaffolding: story-bounded masking, continuation-aware weighting hooks, and a staged E4 config chain are ready to test.
The synthetic backup route should stay separate in reporting: it is a fallback comparison line, not the default first branch to run.
