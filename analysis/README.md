# Refactored analysis directory

The active user-facing analysis surface is now `analysis/go/`.

Use these commands:

```bash
python3 analysis/go/FFT.py DATASET
python3 analysis/go/FFT.py --group MYGROUP
python3 analysis/go/Timeseries.py DATASET
python3 analysis/go/Subtract.py DATASET
python3 analysis/go/Subtract.py --group MYGROUP
python3 analysis/go/ClickPeakFind.py SPECTRASAVE
python3 analysis/go/SpectrasaveView.py SPECTRASAVE
python3 analysis/go/Wavefunctions.py DATASET PEAKSNAME
python3 analysis/go/MakeGroup.py DATASET1 DATASET2 MYGROUP
```

Directory roles:

- `go/` — active user-facing command surface
- `tools/` — data loading, signal processing, catalog, and helper logic
- `plotting/` — reusable plotting helpers
- `viz/` — implementation modules still used by `go/` during the refactor
- `.archive/` — preserved old entry scripts, experiments, and no-longer-primary tools

Notes:

- The loader understands both old flat component datasets like `DATASET_x` and
  the new tracking layout under `track/data/DATASET/components/x`.
- `analysis/viz/` is no longer the recommended place to start from as a user.
- Older exploratory and one-off `viz/` scripts have been moved into
  `analysis/.archive/viz/`.
