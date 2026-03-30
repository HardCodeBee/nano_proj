# Goal
- Current goal: maintain repository-local context for Task 2 / Task 3 ROCStories story-generation work.
- This round scope: re-read Task 2 context plus raw source, sync any newly discovered experiment state, and keep credential handling notes aligned with repo constraints.
- Non-goals: do not change the course evaluation chain; do not rewrite Task 1 historical conclusions; do not treat old `r19` as a fair baseline on the new `ROC val`.

# Hard Constraints
- `<= 32M`
- Do not break the default evaluation chain.
- Keep default text preprocessing compatibility.
- Task 2 must still serve story generation.
- Task 3 must submit the best improved story checkpoint.
- Do not write raw API keys into repo docs or source files; judge credentials must stay in environment variables or CLI flags.
- Treat `eval.py`, `hf_load.py`, and default load/upload compatibility paths as stable unless there is a very strong reason.
- `archive/task1/` is a historical reference area; do not rewrite its conclusions.
- `out-rocstories-remote-r19` may be cited only with a comparability / leakage caveat on the new `ROC val`.

# Protocol
- Current split policy: ROCStories uses `train / val / locked_test`; `val` is a 5% holdout from official train, `locked_test` is the official public test.
- Current split sizes from `data/rocstories/dataset_stats.json`: `train = 74,601`, `val = 3,927`, `locked_test = 19,633` stories with split seed `2027`.
- Current comparison rule: daily Task 2 comparison is only among runs trained under the new split policy.
- Current comparison rule: only checkpoints that have gone through `scripts/task2_generate_and_score.py` and been written to `out-task2/results.csv` count as scored daily comparisons; unscored local checkpoints do not change the bar yet.
- Evaluation script and prompts: `scripts/task2_generate_and_score.py` with `prompts/task2_eval_prompts.txt`.
- Unfair-comparison caveat: historical `r19` was trained on the full official train set, so its score on the new `ROC val` is leaky and not a fair apples-to-apples baseline.

# Key Facts
- Confirmed: E1 was the first genuinely useful Task 2 branch.
- Confirmed: `e4-posteot-mask-openai` is still the current best scored daily `ROC val` checkpoint.
- Confirmed: E3 used the corrected prompt-adherence synthetic data, but did not clearly beat E2.
- Confirmed from source: `train.py` now supports story-bounded supervision in `prepare_targets_and_weights`; `mask_after_story_end = True` masks spillover targets past the current story `EOT`, and `loss_mode = continuation_weighted` applies weighted CE over the valid opening/continuation region.
- Confirmed from source: `train.py` now supports `init_from = warmstart_path`; `warmstart_model_from_checkpoint` can expand depth by copying the last source block forward and prefix-copying positional embeddings when shapes differ.
- Confirmed from source: `data/rocstories/prepare.py --metadata-only` can rebuild the missing ROCStories `train_*` / `val_*` story metadata arrays from existing token streams without re-downloading the dataset.
- Confirmed from source: `scripts/task2_generate_and_score.py` now accepts judge credentials from `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, or `OPENAI_API_KEY`, with `OPENAI_BASE_URL` / `OPENAI_MODEL` fallback support for OpenAI-compatible scoring endpoints.
- Confirmed from `out-task2/results.csv`: `e4-posteot-mask-openai` improved to `avg_loss / ppl = 3.165 / 23.70` with local mean judge `2.1`, outperforming both E2 and the later E4 stages.
- Confirmed from `out-task2/results.csv`: the heavier E4 stages did not beat Step 1; `e4-continuation-weighted-openai` returned to judge `2.0`, while `e4-ending-boost-openai` and `e4-recovery-openai` dropped to `1.9`.
- Confirmed from `out-task2/results.csv`: `e5-synth-continuation-weighted-openai` reached the highest local judge so far at `2.4`, but with a very large token-level regression to `avg_loss / ppl = 3.796 / 44.51`.
- Confirmed from `out-task2/results.csv`: `e5-recovery-openai` partially repaired token-level fit to `3.234 / 25.39`, but its local judge fell back to `2.0`, so E5 did not produce a clean overall win over `e4-posteot-mask-openai`.
- Confirmed from `out-task2/results.csv`: `e6-mixed-masked-openai` scored `3.173 / 23.88 / 2.0`, so restoring broader mixed real-ROC coverage did not preserve the E4 Step-1 gain.
- Confirmed from `out-task2/results.csv`: `e6-gentle-continuation-openai` scored `3.177 / 23.98 / 2.0`, so a milder continuation-weighted follow-up also failed to improve over `e4-posteot-mask-openai`.
- Confirmed from `out-task2/results.csv`: `e7-warmstart-depth7-bs128-openai` scored `3.206 / 24.67 / 2.0`, so the near-capacity warm-start expansion did not improve either token-level fit or local judged quality under the current recipe.
- Confirmed from remote checkpoint inspection: the returned checkpoints finished at `iter_num = 16500` for E6 Step 1, `17500` for E6 Step 2, and `6000` for E7, with checkpoint `best_val_loss` values `3.4166`, `3.4295`, and `3.4504` respectively.
- Confirmed from source: both `config/train_rocstories_task2_e4_recovery.py` and `config/train_rocstories_task2_e5_recovery.py` disable `mask_after_story_end` during their recovery stage, so the earlier recovery recipes did not preserve the clearest story-boundary cleanup win from E4 Step 1.
- Confirmed from repo state: a new follow-up config `config/train_rocstories_task2_e5_masked_recovery.py` now exists to test a short real-ROC recovery from `e5-synth-continuation-weighted` while keeping `mask_after_story_end = True`.
- Confirmed from model sizing: the `e7-warmstart-depth7-bs128` config is about `31.71M` non-embedding parameters under the current `model.py`, so it stays within the `<= 32M` course cap.
- Confirmed: active Task 2 documentation is now intentionally limited to `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/results.csv`, and `out-task2/decision_log.md`; the old `task2_current_status.md` / `task2_experiments.md` placeholders were removed to avoid duplicate maintenance.
- Known limitation: local judge scores remain stuck around `2.0`; improving `ppl` has been easier than improving judged story quality.
- Workflow rule: for every Task 2 / Task 3 turn, first classify the request, then read `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, and the relevant raw source files before implementation, debugging, evaluation, or report writing.
- Workflow rule: if a future request is not Task 2 / Task 3, use `docs/codex_context.md` as the general repository context file and create it from a minimal template if it is missing.
- Update rule: when facts change, update only the affected sections; keep new metrics in `out-task2/results.csv` first, then sync the short summary here and any report-ready interpretation in `out-task2/task2_working_notes.md`.
- Remote execution workflow: local Task 2 code or config changes must be committed and pushed to GitHub first; remote training machines should `git pull` before running and must not assume uncommitted local files exist remotely.

# Important Files
- Path: `out-task2/codex_context.md`
  Role: main Task 2 / Task 3 context file
  Related items: none
- Path: `out-task2/task2_working_notes.md`
  Role: report-oriented narrative notes
  Related items: none
- Path: `out-task2/decision_log.md`
  Role: append-only decision log
  Related items: none
- Path: `out-task2/results.csv`
  Role: structured experiment results table
  Related items: written by `scripts/task2_generate_and_score.py`
- Path: `docs/codex_context.md`
  Role: general repository context file for non-Task-2 / non-Task-3 work
  Related items: create from minimal template if missing
- Path: `data/rocstories/dataset_stats.json`
  Role: authoritative split-policy and split-size metadata for `train / val / locked_test`
  Related items: written by `data/rocstories/prepare.py`
- Path: `train.py`
  Role: main training loop
  Related items: `get_batch`, `choose_start_indices`, `prepare_targets_and_weights`, `compute_loss_for_batch`, `sampling_mode`, `loss_mode`, `resume_ckpt_path`, `warmstart_model_from_checkpoint`, `warmstart_copy_last_block`
- Path: `data/rocstories/prepare.py`
  Role: ROCStories split generation and story-metadata export
  Related items: `train_story_starts.npy`, `train_story_lengths.npy`, `train_first_sentence_lengths.npy`, `val_story_starts.npy`, `val_story_lengths.npy`, `val_first_sentence_lengths.npy`, `--metadata-only`, `rebuild_metadata_from_bin`
- Path: `data/rocstories_synth/prepare.py`
  Role: synthetic ROCStories-style dataset preparation with continuation metadata
  Related items: `train_first_sentence_lengths.npy`, `val_first_sentence_lengths.npy`
- Path: `scripts/task2_generate_and_score.py`
  Role: unified Task 2 evaluation
  Related items: `run_eval_via_course_script`, `default_judge_api_key`, `default_judge_base_url`, `default_judge_model`, fixed prompts, local judge proxy
- Path: `scripts/generate_rocstories_synthetic.py`
  Role: E3 synthetic-data generation
  Related items: `source_opening`, `opening_alignment_too_low_*`
- Path: `config/train_rocstories_task2_e4_posteot_mask.py`
  Role: Step-1 spillover-masking ablation config
  Related items: `mask_after_story_end`, `sampling_mode = story_start`
- Path: `config/train_rocstories_task2_e4_continuation_weighted.py`
  Role: Step-2 continuation-weighted polish config
  Related items: `loss_mode = continuation_weighted`, `prompt_weight`, `continuation_weight`
- Path: `config/train_rocstories_task2_e4_ending_boost.py`
  Role: Step-3 ending-boost config
  Related items: `ending_weight`, `ending_tokens`
- Path: `config/train_rocstories_task2_e4_recovery.py`
  Role: Step-4 short recovery-mix config
  Related items: `loss_mode = standard`, `sampling_mode = mixed`
- Path: `config/train_rocstories_synth_task2_e5_continuation_weighted.py`
  Role: E5 synthetic backup Stage-1 config
  Related items: `dataset = rocstories_synth`, `loss_mode = continuation_weighted`
- Path: `config/train_rocstories_task2_e5_recovery.py`
  Role: E5 synthetic backup Stage-2 real-ROC recovery config
  Related items: `dataset = rocstories`, `loss_mode = standard`
- Path: `config/train_rocstories_task2_e5_masked_recovery.py`
  Role: E5 synthetic backup Stage-2b masked real-ROC recovery config
  Related items: `resume_ckpt_path = out-task2-e5-synth-continuation-weighted/ckpt.pt`, `sampling_mode = mixed`, `mask_after_story_end = True`
- Path: `config/train_rocstories_task2_e6_mixed_masked.py`
  Role: E6 Step-1 mixed-sampling real-ROC follow-up from `e4-posteot-mask`
  Related items: `resume_ckpt_path = out-task2-e4-posteot-mask/ckpt.pt`, `sampling_mode = mixed`, `mask_after_story_end = True`
- Path: `config/train_rocstories_task2_e6_gentle_continuation.py`
  Role: E6 Step-2 milder continuation-weighted follow-up
  Related items: `resume_ckpt_path = out-task2-e6-mixed-masked/ckpt.pt`, `continuation_weight = 1.25`, `ending_weight = 1.1`
- Path: `config/train_rocstories_task2_e7_warmstart_depth7_bs128.py`
  Role: E7 near-capacity warm-start expansion config
  Related items: `init_from = warmstart_path`, `n_layer = 7`, `block_size = 128`

# Latest Runs
- `run_name`: `e1-tinystories-to-rocstories-openai`
  `out_dir`: `out-task2-e1-tinystories-to-rocstories`
  `dataset_recipe`: `TinyStories subset -> ROCStories curriculum`
  `avg_loss / ppl`: `3.181 / 24.07`
  `judge score`: `2.0`
  Notes: first useful Task 2 branch
- `run_name`: `e2-storyaware-polish-openai`
  `out_dir`: `out-task2-e2-storyaware-polish`
  `dataset_recipe`: `TinyStories subset -> ROCStories + story-aware ROC polish`
  `avg_loss / ppl`: `3.176 / 23.95`
  `judge score`: `2.0`
  Notes: clean modest gain over E1; later surpassed by `e4-posteot-mask-openai`
- `run_name`: `e3-synth-distill-openai`
  `out_dir`: `out-task2-e3-synth-distill`
  `dataset_recipe`: `Synthetic prompt-aligned distillation + ROCStories polish`
  `avg_loss / ppl`: `3.176 / 23.96`
  `judge score`: `2.0`
  Notes: confirmed corrected prompt-adherence synthetic data; not clearly better than E2
- `run_name`: `e4-posteot-mask-openai`
  `out_dir`: `out-task2-e4-posteot-mask`
  `dataset_recipe`: `ROCStories story-start polish + post-EOT mask`
  `avg_loss / ppl`: `3.165 / 23.70`
  `judge score`: `2.1`
  Notes: current best daily `ROC val` checkpoint; strongest E4 result
- `run_name`: `e4-continuation-weighted-openai`
  `out_dir`: `out-task2-e4-continuation-weighted`
  `dataset_recipe`: `ROCStories story-bounded continuation weighting`
  `avg_loss / ppl`: `3.166 / 23.72`
  `judge score`: `2.0`
  Notes: roughly held the Step-1 token-level gain, but did not improve judged story quality
- `run_name`: `e4-ending-boost-openai`
  `out_dir`: `out-task2-e4-ending-boost`
  `dataset_recipe`: `ROCStories continuation weighting + ending boost`
  `avg_loss / ppl`: `3.172 / 23.85`
  `judge score`: `1.9`
  Notes: regressed vs. Step 1 and E2
- `run_name`: `e4-recovery-openai`
  `out_dir`: `out-task2-e4-recovery`
  `dataset_recipe`: `ROCStories continuation weighting + ending boost + short recovery mix`
  `avg_loss / ppl`: `3.172 / 23.86`
  `judge score`: `1.9`
  Notes: recovery did not recover the Step-1 judged-quality gain
- `run_name`: `e5-synth-continuation-weighted-openai`
  `out_dir`: `out-task2-e5-synth-continuation-weighted`
  `dataset_recipe`: `Synthetic continuation-aware polish`
  `avg_loss / ppl`: `3.796 / 44.51`
  `judge score`: `2.4`
  Notes: highest local judge so far, but token-level regression is severe
- `run_name`: `e5-recovery-openai`
  `out_dir`: `out-task2-e5-recovery`
  `dataset_recipe`: `Synthetic continuation-aware polish + short real ROC recovery`
  `avg_loss / ppl`: `3.234 / 25.39`
  `judge score`: `2.0`
  Notes: recovery improved over raw E5 Stage 1 `ppl`, but did not preserve the judge gain
- `run_name`: `e6-mixed-masked-openai`
  `out_dir`: `out-task2-e6-mixed-masked`
  `dataset_recipe`: `ROCStories mixed masked continuation from e4-posteot-mask`
  `avg_loss / ppl`: `3.173 / 23.88`
  `judge score`: `2.0`
  Notes: broader mixed sampling lost the E4 Step-1 edge; no new failure flags, but no quality gain either
- `run_name`: `e6-gentle-continuation-openai`
  `out_dir`: `out-task2-e6-gentle-continuation`
  `dataset_recipe`: `ROCStories mixed masked continuation plus gentle continuation weighting`
  `avg_loss / ppl`: `3.177 / 23.98`
  `judge score`: `2.0`
  Notes: mild continuation weighting did not recover the E4 Step-1 advantage
- `run_name`: `e7-warmstart-depth7-bs128-openai`
  `out_dir`: `out-task2-e7-warmstart-depth7-bs128`
  `dataset_recipe`: `ROCStories warm-started 7-layer 128-context expansion from e4-posteot-mask`
  `avg_loss / ppl`: `3.206 / 24.67`
  `judge score`: `2.0`
  Notes: near-capacity expansion underperformed E4 on both token fit and local judged quality

# Current Status
- Completed: E1 curriculum, E2 story-aware polish, E3 corrected synthetic-distillation implementation and evaluation; explicit repository-local context workflow was recorded for Task 2 / Task 3.
- Completed this round: feedback-aligned continuation-aware training scaffolding was added to `train.py`, `data/rocstories/prepare.py`, and `data/rocstories_synth/prepare.py`, plus four sequential E4 configs.
- Completed this round: the full four-stage E4 main line was trained and evaluated on `ROC val`; only `e4-posteot-mask-openai` clearly improved over E2.
- Completed this round: the E5 synthetic backup route was trained and evaluated end to end.
- Completed this round: `scripts/task2_generate_and_score.py` was updated so remote scoring can fall back to `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` instead of requiring `QWEN_API_KEY`.
- Completed this round: raw source was re-read against the context docs; pending E6 / E7 branches were resynced, and a leaked API key example in `instruction/FurtherInstructions.txt` was removed.
- Completed this round: local ROCStories story metadata was rebuilt from the existing `train.bin` / `val.bin` streams, and the updated E6 / E7 configs now pass local CPU loader sanity checks with the new `warmstart_path` support.
- Completed this round: the remote E6 and E7 follow-up runs were pulled back locally and synced into the fixed Task 2 results table.
- Completed this round: a new masked real-data E5 recovery config was added so the next synthetic-salvage test can keep story-boundary masking on during recovery.
- In progress: decide whether E5 Stage 1's higher local judge merits shortlist attention, and whether masked real-data recovery can preserve any of that gain without keeping the large `ppl` penalty.
- Current blocker: no scored run has yet improved both local judged quality and token-level fit beyond `e4-posteot-mask-openai`; real-data branches remain stuck around judge `2.0-2.1`, and `ppl ≈ 20` is still far away.

# Decisions
- 2026-03-30 | Use new `ROC val` for daily Task 2 comparison | Avoid daily tuning on public test | `data/rocstories/prepare.py`, `scripts/task2_generate_and_score.py`
- 2026-03-30 | Keep E2 as current best daily `ROC val` checkpoint | Small but real token-level improvement over E1 | `config/train_rocstories_task2_e2_storyaware_polish.py`
- 2026-03-30 | Refocus E3 synthetic data toward opening adherence | Final scoring is closer to prompt-conditioned good-story generation than surface ROC-style imitation | `scripts/generate_rocstories_synthetic.py`, `prompts/task2_rocstyle_rewrite_prompt.txt`
- 2026-03-30 | Adopt explicit repository-local externalized context workflow | Reduce reliance on conversation history and force each new turn to re-read context plus relevant source files before work | `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/decision_log.md`, `docs/codex_context.md`
- 2026-03-30 | Implement the feedback-aligned E4 training path in code before new runs | The highest-value next test is to isolate post-`EOT` spillover and then move to continuation-weighted supervision instead of changing source data again | `train.py`, `data/rocstories/prepare.py`, `data/rocstories_synth/prepare.py`, `config/train_rocstories_task2_e4_*.py`
- 2026-03-30 | Keep E5 as a separate synthetic backup line rather than mixing it into E4 | The main uncertainty is still on real ROCStories; synthetic continuation-aware polish should stay as a fallback comparison, not the first branch to run | `config/train_rocstories_synth_task2_e5_continuation_weighted.py`, `config/train_rocstories_task2_e5_recovery.py`
- 2026-03-31 | Let the Task 2 scorer fall back to `OPENAI_*` env vars for judge calls | Remote runs were failing when only `OPENAI_API_KEY` was exported even though the judge endpoint is OpenAI-compatible | `scripts/task2_generate_and_score.py` | Keep secrets in env vars only; do not persist raw keys in repo docs
- 2026-03-31 | Promote `e4-posteot-mask-openai` to the current best daily checkpoint and treat later E4 stages as non-winning ablations | Step 1 improved both `ppl` and local judge, while continuation weighting / ending boost / recovery did not extend the gain | `out-task2/results.csv`, `out-task2/codex_context.md`, `out-task2/task2_working_notes.md` | Keep E5 as the next comparison line, but compare it to E4 Step 1 rather than to E2 alone
- 2026-03-31 | Treat E5 as a mixed-result comparison line rather than the new default winner | Stage 1 produced the best local judge (`2.4`) but with extreme `ppl` regression; Stage 2 improved `ppl` but lost the judge gain | `out-task2/results.csv`, `out-task2/codex_context.md`, `out-task2/task2_working_notes.md` | Keep `e4-posteot-mask-openai` as the clean balanced leader until a branch beats it on both quality and stability
- 2026-03-31 | Remove frozen duplicate Task 2 docs after E4/E5 sync | Keeping placeholder files around was causing avoidable confusion about which documents were active | `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/results.csv`, `out-task2/decision_log.md`, `archive/task1/README.md` | Active Task 2 docs are now only the context file, working notes, results table, and decision log
- 2026-03-31 | Keep E6 / E7 out of the daily comparison table until they pass through the fixed Task 2 scorer | Local checkpoints and drafted configs exist, but the apples-to-apples comparison rule is defined by `scripts/task2_generate_and_score.py` plus `out-task2/results.csv` | `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/results.csv` | `e4-posteot-mask-openai` remains the scored comparison bar for now
- 2026-03-31 | Remove the raw API key example from `instruction/FurtherInstructions.txt` | Repository docs and source must not contain live secrets; judge credentials belong in env vars or CLI flags only | `instruction/FurtherInstructions.txt`, `out-task2/codex_context.md`, `out-task2/decision_log.md` | Use placeholders in docs, never committed keys
- 2026-03-31 | Treat GitHub push/pull as the required bridge between local edits and remote training | Remote shells only see committed and pushed files, so remote execution instructions must include a local push step before any remote `git pull` + train sequence | `out-task2/codex_context.md`, `out-task2/decision_log.md` | Do not reference configs that exist only in the local worktree
- 2026-03-31 | Keep `e4-posteot-mask-openai` as the real-data leader after scoring E6 and E7 | `e6-mixed-masked`, `e6-gentle-continuation`, and `e7-warmstart-depth7-bs128` all returned judge `2.0` and worse `ppl` than E4 Step 1 | `out-task2/results.csv`, `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/decision_log.md` | Mixed sampling and near-capacity expansion did not extend the E4 gain under the current protocol
- 2026-03-31 | Prioritize an E5 masked real-data recovery follow-up over rerunning E6/E7 | E6 and E7 already failed under the fixed scorer, while both older recovery recipes turned `mask_after_story_end` off and may have washed out the one clearly helpful supervision cleanup | `config/train_rocstories_task2_e5_masked_recovery.py`, `out-task2/codex_context.md`, `out-task2/decision_log.md` | Test whether synthetic judge-side gains can be partially retained without reintroducing cross-story spillover

# Verification
- Already run: `data/rocstories/prepare.py`, E1 / E2 / E3 training branches, the full four-stage E4 line, `task2_generate_and_score.py`, local `py_compile` on the new E4-related Python files, local `py_compile` on the updated Task 2 scorer env-fallback patch, source reread of `train.py`, `model.py`, `data/rocstories/prepare.py`, `data/rocstories_synth/prepare.py`, `eval.py`, `scripts/task2_generate_and_score.py`, and the active Task 2 configs
- Result: `e4-posteot-mask-openai` reached `3.165 / 23.70 / 2.1`; `e4-continuation-weighted-openai` reached `3.166 / 23.72 / 2.0`; `e4-ending-boost-openai` reached `3.172 / 23.85 / 1.9`; `e4-recovery-openai` reached `3.172 / 23.86 / 1.9`
- Result: `e5-synth-continuation-weighted-openai` reached `3.796 / 44.51 / 2.4`; `e5-recovery-openai` reached `3.234 / 25.39 / 2.0`
- Result: `data/rocstories/prepare.py --metadata-only` rebuilt the missing `train_*` / `val_*` metadata arrays for `74,601` train and `3,927` val stories from the existing local token streams.
- Result: `e6-mixed-masked-openai` reached `3.173 / 23.88 / 2.0`; `e6-gentle-continuation-openai` reached `3.177 / 23.98 / 2.0`; `e7-warmstart-depth7-bs128-openai` reached `3.206 / 24.67 / 2.0`.
- Result: the returned remote checkpoints finished at `iter_num = 16500`, `17500`, and `6000` for E6 Step 1, E6 Step 2, and E7, with checkpoint `best_val_loss` values `3.4166`, `3.4295`, and `3.4504`.
- Result: the planned `e7-warmstart-depth7-bs128` configuration stayed within the cap at about `31.71M` non-embedding parameters under the current `model.py`, but that capacity increase alone did not improve the fixed-protocol metrics.
- Unverified risk: E5 Stage 1's `2.4` local judge may reflect proxy-judge preference or synthetic-distribution shift rather than a robust overall story-quality gain; it remains the only line that materially moved the judge, but at an unacceptable token-level cost so far.
- Unverified risk: even with `mask_after_story_end = True`, a masked real-data recovery may still erase the synthetic judge gain entirely if the root issue is the synthetic distribution itself rather than the recovery recipe.

# Next Steps
- Next 1: on every future Task 2 / Task 3 turn, read this file and `out-task2/task2_working_notes.md` before new work.
- Next 2: after reading context files, re-read the relevant raw source/config/eval files before each new experiment, bug fix, or report update.
- Next 3: use `e4-posteot-mask-openai` as the new daily comparison bar for future real-data Task 2 work.
- Next 4: inspect why `story_start + post-EOT mask` worked better than the broader mixed-sampling E6 variants; the current evidence suggests the gain was fragile to sampling changes.
- Next 5: run and fixed-score `e5-masked-recovery` to test whether keeping story-boundary masking on during real-data recovery preserves any of the E5 Stage-1 judge gain.
- Next 6: if `e5-masked-recovery` also fails cleanly, avoid treating wider context or one-step capacity expansion as a default fix; any next branch should have a more targeted hypothesis than the current E7 recipe.
