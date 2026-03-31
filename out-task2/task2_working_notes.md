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
- Submission-safe decoding sweeps were then checked as a zero-`ppl`-risk option, but they did not reveal a better frontier; the best E4 setting only tied the existing baseline.
- A ROC-native synthetic regeneration route was the only partly encouraging new branch: it reduced the old E5 Stage-1 `ppl` collapse while keeping a judge score of `2.2`, but its masked recovery still fell back to judge `2.0`.
- Two more training-target ideas were also tested directly: full-story prefix-to-continuation batching and masked annealing. Neither beat the simpler E4 Step-1 checkpoint.
- A new from-scratch `storymix_v1` mainline then tested a different hypothesis: keep the Task 1 model skeleton fixed, rebuild the data regime around narrative-only story pretraining, then retarget on ROCStories and finish with a conservative continuation polish. That line produced a weak-but-real quality signal in Stage C (`judge = 2.2`), but still regressed badly on `ppl`, so it is a clue rather than the new default winner.

## What is safe to claim

- The exploration was real, multi-stage, and evidence-based.
- Better data and better sampling improved token-level fit.
- A simple training-target cleanup mattered: masking spillover after the story `EOT` helped more than the heavier continuation-weighted stages that followed.
- Synthetic continuation-aware polish can raise the local judge, but the current E5 version did so in an unstable way that badly hurt token-level fit.
- Keeping story-boundary masking on during E5 recovery was not enough to turn the synthetic line into a clean winner.
- Moving the synthetic source closer to ROCStories looks healthier than the older TinyStories-style synthetic source, but the present recovery recipe still gives up the judge-side gain.
- For this small model, improving `ppl` turned out to be easier than improving judged story quality.
- The strongest practical checkpoint so far is `e4-posteot-mask-openai`, because it improved both token-level fit and the local judge without needing the later E4 stages.
- The E6 and E7 follow-ups strengthen that conclusion: neither broader mixed sampling nor the current 7-layer warm-start recipe improved over E4 Step 1.
- The decoding sweep result also strengthens that conclusion: there does not appear to be an easy submission-safe generation-parameter fix waiting in the current `temperature / top_k` grid.
- The new `storymix_v1` line is also safe to claim as a meaningful experiment: it showed that a from-scratch narrative-only pretraining route can learn a strong five-sentence story shell and later recover a local judge score of `2.2`, but the first implementation did not preserve token-level fit well enough to displace the older E4 anchor.

## What should be stated carefully

- Daily Task 2 comparisons should be made only among checkpoints trained and evaluated under the new held-out `ROC val` protocol; `locked_test` is for occasional shortlist checks.
- Old `r19` remains the historical Task 1 reference, but it is not a fair baseline on the new `ROC val`.
- E3 should not be oversold. It was a meaningful aggressive experiment, but it did not clearly beat E2.
- The later E4 stages should also not be oversold. Continuation weighting, ending boost, and short recovery were informative ablations, but they did not beat the simpler Step-1 masking result.
- E5 also needs careful framing. Stage 1 may be useful as a clue about what the judge likes, but it is not a clean winner because its `ppl` regressed sharply; Stage 2 recovered some fit but lost the quality gain.
- The masked E5 recovery line, including its small LR sweep, is now complete and should be framed as an informative non-winner rather than a pending rescue path.
- E6 and E7 should now be framed as negative or neutral follow-ups, not pending branches. They were useful ablations, but they did not beat `e4-posteot-mask-openai`.
- The ROC-native synthetic pilot should be framed carefully too: it was healthier than the old synthetic Stage 1, but it still did not produce a recovery-stage win.
- Prefix-to-continuation and masked annealing are now completed negative pilots, not pending ideas.
- The first `storymix_v1` pilot should also be framed carefully: it is not a win over E4, because its best scored stage traded quality signal for a large `ppl` regression.
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
The masked recovery result strengthens that interpretation: even when the recovery stage keeps the post-`EOT` cleanup, the local judge gain still disappears once the model is pulled back toward real ROCStories.
The ROC-native synthetic pilot keeps one door open: it reduced the old Stage-1 `ppl` damage from `44.51` to `33.79` while still reaching judge `2.2`, so synthetic source mismatch looks real rather than incidental. But the masked ROC-native recovery still fell back to judge `2.0`, which means the current recovery recipe remains the main bottleneck on that line.
E6 adds a third clue: the E4 gain was not robust to restoring broader mixed-sampling coverage, so part of that gain may depend on the story-start-heavy training distribution rather than on a universally better objective.
E7 adds a fourth clue: a near-capacity depth/context expansion by itself is not enough under the current recipe and budget; the bottleneck is not solved just by making the model slightly larger.
The decoding sweep result matters mostly as a negative control: the current judged-quality ceiling does not look like a simple `sample_params.json` mistake, because the tested frontier only reproduced the existing E4 score.
That specific masking question is now partly answered: turning the mask back on helped only marginally on token fit and did not recover judged quality, so the dominant issue appears to be the synthetic distribution shift itself rather than the missing mask alone.
The E8 pilots reinforce the broader lesson: making the continuation target more explicit or annealing toward broader coverage did not outperform the simpler post-`EOT` masking fix once everything was scored under the same protocol.
The new `storymix_v1` pilot adds a fifth clue. A fresh narrative-only data regime can teach the model a cleaner five-sentence story shape from scratch, and a conservative continuation-polish stage can move the judge back up to `2.2`. But because the ROC-only adaptation stage did not first produce a strong ROC-aligned base, the final Stage C result still paid too much in `ppl`. That suggests the data-regime idea is not obviously wrong; the current Stage A / Stage B handoff is probably the main bottleneck on this new line.

## If writing the report now

Main Task 2 arc:

1. Start from a frozen historical ROCStories reference.
2. Test curriculum learning with TinyStories.
3. Add story-aware polish.
4. Try a more aggressive prompt-aligned synthetic-distillation branch.
5. Test story-bounded masking and continuation-weighted supervision on real ROCStories.
6. Test broader mixed-sampling and near-capacity warm-start follow-ups.
7. Test a fresh `storymix_v1` from-scratch mainline built around narrative-only pretraining, ROC-only adaptation, and conservative continuation polish.
8. Conclude that the best practical checkpoint still came from the simple post-`EOT` masking fix, while token-level improvements remained easier to obtain than consistent judged story-quality gains.

## Next-step note

If experimentation continues, the real-data anchor should still be `e4-posteot-mask-openai`.
This direction has now been tested more fully: story-bounded masking worked best, while the heavier continuation-weighted stages, the broader mixed-sampling follow-up, the current near-capacity warm-start expansion, the prefix-to-continuation pilot, and masked annealing all failed to surpass it.
The synthetic backup route should still stay separate in reporting: it surfaced an interesting judge-side gain, and the ROC-native synthetic source looked healthier than the old one, but the real-data bar to beat remains `e4-posteot-mask-openai` until that gain can be retained without the recovery-stage collapse.
The new `storymix_v1` line should also stay separate in reporting from the older E4 winner. It is the most relevant current clue if the project wants to argue for a different data regime, but the first pilot should be presented as an informative non-winner rather than as the new default path.
The next execution plan should no longer treat masked E5 recovery, decoding sweeps on old non-winners, E8 prefix-to-continuation, or masked annealing as priority branches. The only lines that still look mildly worth another follow-up are ROC-native synthetic with a more conservative recovery design and `storymix_v1` with a healthier Stage A / Stage B handoff.
