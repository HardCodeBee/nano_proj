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
- Confirmed: E2 is the current best daily `ROC val` checkpoint.
- Confirmed: E3 used the corrected prompt-adherence synthetic data, but did not clearly beat E2.
- Confirmed from source: `train.py` story-aware sampling changes window start positions, but the training loss is still uniform next-token CE over the whole contiguous window; there is no post-`EOT` masking in the current loop.
- Confirmed from source: `scripts/task2_generate_and_score.py` now accepts judge credentials from `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, or `OPENAI_API_KEY`, with `OPENAI_BASE_URL` / `OPENAI_MODEL` fallback support for OpenAI-compatible scoring endpoints.
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

# Current Status
- Completed: E1 curriculum, E2 story-aware polish, E3 corrected synthetic-distillation implementation and evaluation; explicit repository-local context workflow was recorded for Task 2 / Task 3.
- Completed this round: feedback-aligned continuation-aware training scaffolding was added to `train.py`, `data/rocstories/prepare.py`, and `data/rocstories_synth/prepare.py`, plus four sequential E4 configs.
- Completed this round: the E5 synthetic backup route was added as a separate two-stage config chain.
- Completed this round: `scripts/task2_generate_and_score.py` was updated so remote scoring can fall back to `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` instead of requiring `QWEN_API_KEY`.
- In progress: regenerate local dataset metadata and run the E4 main line on remote GPU first, with E5 held as the synthetic backup route if E4 stalls.
- Current blocker: no branch has yet produced a clear jump in prompt-conditioned story quality.

# Decisions
- 2026-03-30 | Use new `ROC val` for daily Task 2 comparison | Avoid daily tuning on public test | `data/rocstories/prepare.py`, `scripts/task2_generate_and_score.py`
- 2026-03-30 | Keep E2 as current best daily `ROC val` checkpoint | Small but real token-level improvement over E1 | `config/train_rocstories_task2_e2_storyaware_polish.py`
- 2026-03-30 | Refocus E3 synthetic data toward opening adherence | Final scoring is closer to prompt-conditioned good-story generation than surface ROC-style imitation | `scripts/generate_rocstories_synthetic.py`, `prompts/task2_rocstyle_rewrite_prompt.txt`
- 2026-03-30 | Adopt explicit repository-local externalized context workflow | Reduce reliance on conversation history and force each new turn to re-read context plus relevant source files before work | `out-task2/codex_context.md`, `out-task2/task2_working_notes.md`, `out-task2/decision_log.md`, `docs/codex_context.md`
- 2026-03-30 | Implement the feedback-aligned E4 training path in code before new runs | The highest-value next test is to isolate post-`EOT` spillover and then move to continuation-weighted supervision instead of changing source data again | `train.py`, `data/rocstories/prepare.py`, `data/rocstories_synth/prepare.py`, `config/train_rocstories_task2_e4_*.py`
- 2026-03-30 | Keep E5 as a separate synthetic backup line rather than mixing it into E4 | The main uncertainty is still on real ROCStories; synthetic continuation-aware polish should stay as a fallback comparison, not the first branch to run | `config/train_rocstories_synth_task2_e5_continuation_weighted.py`, `config/train_rocstories_task2_e5_recovery.py`
- 2026-03-31 | Let the Task 2 scorer fall back to `OPENAI_*` env vars for judge calls | Remote runs were failing when only `OPENAI_API_KEY` was exported even though the judge endpoint is OpenAI-compatible | `scripts/task2_generate_and_score.py` | Keep secrets in env vars only; do not persist raw keys in repo docs

# Verification
- Already run: `data/rocstories/prepare.py`, E1 / E2 / E3 training branches, `task2_generate_and_score.py`, local `py_compile` on the new E4-related Python files, local `py_compile` on the updated Task 2 scorer env-fallback patch
- Result: E1 was useful, E2 is the best small improvement, E3 corrected still did not beat local judge score `2.0`
- Unverified risk: with `block_size = 96` and ROCStories mean length around `52`, many story-start windows may include post-`EOT` spill into the next story; this has not yet been isolated by an ablation.

# Next Steps
- Next 1: on every future Task 2 / Task 3 turn, read this file and `out-task2/task2_working_notes.md` before new work.
- Next 2: after reading context files, re-read the relevant raw source/config/eval files before each new experiment, bug fix, or report update.
- Next 3: prioritize a fast ablation that masks target tokens after the first `EOT` in story-start windows before moving to heavier continuation-weighted training changes.
- Next 4: regenerate `rocstories` and `rocstories_synth` local artifacts so the new `*_first_sentence_lengths.npy` files exist before remote training.
- Next 5: keep E4 and E5 remote run instructions separate so the synthetic backup route is only used if the real-data line stalls or clearly underperforms.
