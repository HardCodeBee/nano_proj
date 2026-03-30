# Task 2 Working Notes

## Purpose

This file keeps short narrative notes that are useful for later report writing.
It should stay lighter than `out-task2/codex_context.md` and `out-task2/results.csv`, and focus on interpretation rather than exhaustive inventory.

## Current report-ready storyline

- Task 1 established a valid ROCStories baseline, but the project goal moved to improving story generation for Task 2 / Task 3.
- E1 showed that a TinyStories -> ROCStories curriculum can help: it was the first clearly useful Task 2 result.
- E2 added story-aware ROCStories polish and gave a clean but modest gain over E1, but it was later surpassed by the simpler E4 post-`EOT` masking step.
- E3 tested a more aggressive prompt-aligned synthetic-distillation branch. It was novel and correctly targeted opening adherence, but it still did not produce a decisive judged-quality jump.
- E4 finally produced a clearer signal: the simple post-`EOT` masking ablation helped, but the heavier continuation-weighted follow-up stages did not extend that gain.
- E5 added a synthetic continuation-aware backup line. It produced the highest local judge score so far in Stage 1, but with a severe `ppl` regression, and the recovery stage gave up that judge gain while only partly repairing token-level fit.
- E6 and E7 then tested two different follow-up ideas on real ROCStories: broader mixed sampling and a near-capacity warm-start expansion. Neither beat the simpler E4 Step-1 checkpoint.
- The next worthwhile follow-up is no longer another E6/E7-style branch. The sharper test is whether E5's proxy-judge gain can survive a real-data recovery that keeps story-boundary masking on.

## What is safe to claim

- The exploration was real, multi-stage, and evidence-based.
- Better data and better sampling improved token-level fit.
- A simple training-target cleanup mattered: masking spillover after the story `EOT` helped more than the heavier continuation-weighted stages that followed.
- Synthetic continuation-aware polish can raise the local judge, but the current E5 version did so in an unstable way that badly hurt token-level fit.
- For this small model, improving `ppl` turned out to be easier than improving judged story quality.
- The strongest practical checkpoint so far is `e4-posteot-mask-openai`, because it improved both token-level fit and the local judge without needing the later E4 stages.
- The E6 and E7 follow-ups strengthen that conclusion: neither broader mixed sampling nor the current 7-layer warm-start recipe improved over E4 Step 1.

## What should be stated carefully

- Daily Task 2 comparisons should be made only among checkpoints trained and evaluated under the new held-out `ROC val` protocol; `locked_test` is for occasional shortlist checks.
- Old `r19` remains the historical Task 1 reference, but it is not a fair baseline on the new `ROC val`.
- E3 should not be oversold. It was a meaningful aggressive experiment, but it did not clearly beat E2.
- The later E4 stages should also not be oversold. Continuation weighting, ending boost, and short recovery were informative ablations, but they did not beat the simpler Step-1 masking result.
- E5 also needs careful framing. Stage 1 may be useful as a clue about what the judge likes, but it is not a clean winner because its `ppl` regressed sharply; Stage 2 recovered some fit but lost the quality gain.
- A new masked E5 recovery should still be framed as a targeted salvage test, not as an already-validated improvement.
- E6 and E7 should now be framed as negative or neutral follow-ups, not pending branches. They were useful ablations, but they did not beat `e4-posteot-mask-openai`.
- The local automatic judge is only a proxy for the final evaluation; it is useful, but not the official private-test scorer.

## Failure pattern summary

- The model can often stay on the general topic of the opening sentence.
- The harder part is sustaining a coherent event chain through the middle of the story.
- Natural endings remain inconsistent.
- Many samples are still bland even when they are not obviously broken.

## Best current interpretation

The main bottleneck is no longer simply dataset choice.
The deeper problem is that the model still struggles with prompt-conditioned continuation: turning an opening sentence into a compact, coherent, naturally ending short story.
E4 suggests one concrete part of that problem was real training noise from cross-story spillover. Cleaning that up helped; more aggressive continuation reweighting did not yet produce an additional win.
E5 suggests a second clue: synthetic continuation-aware training can move the local judge, but the present recipe is too distribution-shifting to keep the model well calibrated on standard token-level fit.
E6 adds a third clue: the E4 gain was not robust to restoring broader mixed-sampling coverage, so part of that gain may depend on the story-start-heavy training distribution rather than on a universally better objective.
E7 adds a fourth clue: a near-capacity depth/context expansion by itself is not enough under the current recipe and budget; the bottleneck is not solved just by making the model slightly larger.
That leaves one particularly concrete unresolved question: whether the old E5 recovery recipe failed partly because it abandoned the post-`EOT` masking cleanup instead of only because the synthetic distribution shift was too strong.

## If writing the report now

Main Task 2 arc:

1. Start from a frozen historical ROCStories reference.
2. Test curriculum learning with TinyStories.
3. Add story-aware polish.
4. Try a more aggressive prompt-aligned synthetic-distillation branch.
5. Test story-bounded masking and continuation-weighted supervision on real ROCStories.
6. Test broader mixed-sampling and near-capacity warm-start follow-ups.
7. Conclude that the best practical checkpoint still came from the simple post-`EOT` masking fix, while token-level improvements remained easier to obtain than consistent judged story-quality gains.

## Next-step note

If experimentation continues, the next useful branch should change the training target more directly around opening-to-story continuation rather than only changing the source corpus again.
This direction has now been tested more fully: story-bounded masking worked best, while the heavier continuation-weighted stages, the broader mixed-sampling follow-up, and the current near-capacity warm-start expansion all failed to surpass it.
The synthetic backup route should still stay separate in reporting: it surfaced an interesting judge-side gain, but the real-data bar to beat remains `e4-posteot-mask-openai` until that gain can be retained without the large `ppl` penalty.
The next execution plan should no longer treat E6 or E7 as pending. Instead, it should explain why the E4 Step-1 gain disappeared under broader sampling and whether the only judge-moving line, E5, can be stabilized without breaking token-level fit, starting with a masked real-data recovery rather than another fresh architecture branch.
