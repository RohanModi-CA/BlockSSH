This directory is for preserved analysis scripts and experiments that are not
part of the main refactored user-facing path.

Policy:

- `analysis/go/` is the active CLI surface for the refactor.
- `analysis/viz/` remains available during migration, but individual scripts can
  move here once their `go/` replacement is stable.
- `analysis/viz/see_fft.py` is now preserved here and replaced in-place by a
  compatibility wrapper that forwards to `see_fft_xya.py`.
- Temporary experiments and one-off diagnostics belong here if they are not
  part of the intended long-term workflow.

Archived under `analysis/.archive/viz/` in this cleanup pass:

- old exploratory FFT / amplitude / localization entrypoints
- cleantest variants
- area / unbarrel utilities
- one-off diagnostics and temporary scripts that are not on the `analysis/go/`
  dependency path
