This directory is for tracking code and prototypes that are intentionally kept
out of the main refactored workflow.

Current policy:

- Bottom black tracking is the only active refactor target.
- Front / non-black tracking is intentionally retained but not being modernized
  in this pass.
- Area-based experiments are preserved here if they do not fit the mainline.
- Old scripts should only be moved here after the new `track/Bottom/` workflow
  replaces them.
- The active bottom workflow lives in `track/Bottom/`.
- Root-level non-bottom scripts that remain outside this archive are preserved
  intentionally and should not be treated as part of the refactored bottom path.

Archived here in the current cleanup:

- the old generic `0/1/2` tracking entrypoints
- the old generic `B0/B1/B2` batch entrypoints
- the old tracking API note
- standalone generic utilities like `debug.py`, `rotate_track1.py`, and
  `convert_mat_to_btp.py`

Nothing in this directory should be treated as the primary user-facing path.
