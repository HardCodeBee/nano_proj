# Goal
- Current goal: maintain repository-local context for Task 2 / Task 3 ROCStories story-generation work.
- This round scope: initialize the externalized-context workflow and record the current confirmed Task 2 state.
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
- Evaluation script and prompts: `scripts/task2_generate_and_score.py` with `prompts/task2_eval_prompts.txt`.
- Unfair-comparison caveat: historical `r19` was trained on the full official train set, so its score on the new `ROC val` is leaky and not a fair apples-to-apples baseline.

# Key Facts
- Confirmed: E1 was the first genuinely useful Task 2 branch.
- Confirmed: `e4-posteot-mask-openai` is the current best daily `ROC val` checkpoint.
- Confirmed: E3 used the corrected prompt-adherence synthetic data, but did not clearly beat E2.
- Confirmed from source: `train.py` story-aware sampling changes window start positions, but the training loss is still uniform next-token CE over the whole contiguous window; there is no post-`EOT` masking in the current loop.
- Confirmed from source: `scripts/task2_generate_and_score.py` now accepts judge credentials from `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, or `OPENAI_API_KEY`, with `OPENAI_BASE_URL` / `OPENAI_MODEL` fallback support for OpenAI-compatible scoring endpoints.
- Confirmed from `out-task2/results.csv`: `e4-posteot-mask-openai` improved to `avg_loss / ppl = 3.165 / 23.70` with local mean judge `2.1`, outperforming both E2 and the later E4 stages.
- Confirmed from `out-task2/results.csv`: the heavier E4 stages did not beat Step 1; `e4-continuation-weighted-openai` returned to judge `2.0`, while `e4-ending-boost-openai` and `e4-recovery-openai` dropped to `1.9`.
- Confirmed from `out-task2/results.csv`: `e5-synth-continuation-weighted-openai` reached the highest local judge so far at `2.4`, but with a very large token-level regression to `avg_loss / ppl = 3.796 / 44.51`.
- Confirmed from `out-task2/results.csv`: `e5-recovery-openai` partially repaired token-level fit to `3.234 / 25.39`, but its local judge fell back to `2.0`, so E5 did not produce a clean overall win over `e4-posteot-mask-openai`.
- Confirmed: active Task 2 documentation is now intentionally limited to `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/results.csv`, and `out-task2/decision_log.md`; the old `task2_current_status.md` / `task2_experiments.md` placeholders were removed to avoid duplicate maintenance.
- Known limitation: local judge scores remain stuck around `2.0`; improving `ppl` has been easier than improving judged story quality.
- Workflow rule: for every Task 2 / Task 3 turn, first classify the request, then read `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, and the relevant raw source files before implementation, debugging, evaluation, or report writing.
- Workflow rule: if a future request is not Task 2 / Task 3, use `docs/codex_context.md` as the general repository context file and create it from a minimal template if it is missing.
- Update rule: when facts change, update only the affected sections; keep new metrics in `out-task2/results.csv` first, then sync the short summary here and any report-ready interpretation in `out-task2/task2_working_notes.md`.

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
  Related items: `get_batch`, `choose_start_indices`, `prepare_targets_and_weights`, `compute_loss_for_batch`, `sampling_mode`, `loss_mode`, `resume_ckpt_path`
- Path: `data/rocstories/prepare.py`
  Role: ROCStories split generation and story-metadata export
  Related items: `train_story_starts.npy`, `train_story_lengths.npy`, `train_first_sentence_lengths.npy`, `val_story_starts.npy`, `val_story_lengths.npy`, `val_first_sentence_lengths.npy`
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
  Notes: current best daily `ROC val` checkpoint
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

# Current Status
- Completed: E1 curriculum, E2 story-aware polish, E3 corrected synthetic-distillation implementation and evaluation; explicit repository-local context workflow was recorded for Task 2 / Task 3.
- Completed this round: feedback-aligned continuation-aware training scaffolding was added to `train.py`, `data/rocstories/prepare.py`, and `data/rocstories_synth/prepare.py`, plus four sequential E4 configs.
- Completed this round: the full four-stage E4 main line was trained and evaluated on `ROC val`; only `e4-posteot-mask-openai` clearly improved over E2.
- Completed this round: the E5 synthetic backup route was trained and evaluated end to end.
- Completed this round: `scripts/task2_generate_and_score.py` was updated so remote scoring can fall back to `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` instead of requiring `QWEN_API_KEY`.
- In progress: decide whether to value E5 Stage 1's higher local judge enough to shortlist it despite the very poor `ppl`, or to keep `e4-posteot-mask-openai` as the cleaner balanced leader.
- Current blocker: no run has yet improved both local judged quality and token-level fit at the same time beyond `e4-posteot-mask-openai`.

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

# Verification
- Already run: `data/rocstories/prepare.py`, E1 / E2 / E3 training branches, the full four-stage E4 line, `task2_generate_and_score.py`, local `py_compile` on the new E4-related Python files, local `py_compile` on the updated Task 2 scorer env-fallback patch
- Result: `e4-posteot-mask-openai` reached `3.165 / 23.70 / 2.1`; `e4-continuation-weighted-openai` reached `3.166 / 23.72 / 2.0`; `e4-ending-boost-openai` reached `3.172 / 23.85 / 1.9`; `e4-recovery-openai` reached `3.172 / 23.86 / 1.9`
- Result: `e5-synth-continuation-weighted-openai` reached `3.796 / 44.51 / 2.4`; `e5-recovery-openai` reached `3.234 / 25.39 / 2.0`
- Unverified risk: with `block_size = 96` and ROCStories mean length around `52`, many story-start windows may include post-`EOT` spill into the next story; this has not yet been isolated by an ablation.

# Next Steps
- Next 1: on every future Task 2 / Task 3 turn, read this file and `out-task2/task2_working_notes.md` before new work.
- Next 2: after reading context files, re-read the relevant raw source/config/eval files before each new experiment, bug fix, or report update.
- Next 3: use `e4-posteot-mask-openai` as the new daily comparison bar for future real-data Task 2 work.
- Next 4: inspect E5 Stage 1 samples closely before deciding whether its higher local judge is meaningful enough to justify shortlist attention despite the poor `ppl`.
- Next 5: if experimentation continues beyond E5, preserve post-`EOT` masking as the baseline fix and look for variants that keep or improve the E5 judge gain without the large token-level regression.
