# Task 1 Archive

This archive groups together Task 1 tuning history and local debugging artifacts so
the active workspace can stay focused on Task 2.

## Active files kept outside the archive

- `config/train_rocstories.py`
- `config/train_rocstories_r19_frozen.py`
- `config/train_tinystories_task2_e1_stage1.py`
- `config/train_rocstories_task2_e1_stage2.py`
- `out-task2/codex_context.md`
- `out-task2/task2_working_notes.md`
- `out-task2/results.csv`
- `out-rocstories-remote-r19/`

## Baseline record

- Best documented Task 1 artifact: `out-rocstories-remote-r19/`
- Public-test result: `avg_loss = 3.216`, `ppl = 24.93`
- Frozen reproduction recipe: `config/train_rocstories_r19_frozen.py`

## Archive contents

- `config_history/`: old Task 1 tuning configs and seed-sweep notes
- `notes/`: Task 1 writeups, drafts, and local debugging notes kept in Git
- `run_history/`: local-only non-active run directories and smoke-test outputs

`run_history/` is intentionally gitignored so the repository stays lightweight.
It is a local parking area, not part of the clean project state that should be
pulled onto the remote GPU workspace.

Nothing here was deleted on purpose; files were grouped to reduce clutter while
preserving the Task 1 paper trail.
