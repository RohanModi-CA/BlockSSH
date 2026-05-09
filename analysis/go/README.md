# Analysis Go Surface

This directory is the new user-facing analysis entry path.

Current commands:

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
python3 analysis/go/RohanTop.py DATASET --component x --tmin 5 --tmax 47
```

Notes:

- `FFT.py` is based on the component-aware `see_fft_xya.py` path.
- grouped `FFT.py` delegates to the averaged-spectrum path by building a temporary
  selection config from the datasets in the named group.
- `Subtract.py` accepts a dataset, a named group, or an explicit `--config-json`.
- dataset and group config synthesis derive bond ids from tracked site labels
  when available, and otherwise fall back to left-to-right local numbering.
- `RohanTop.py` analyzes a chosen `x`, `y`, or `a` component over a manual hit
  window and can batch over `--group` from `MakeGroup.py`.
- The underlying internals still live in `analysis/tools/`, `analysis/plotting/`,
  and the current `analysis/viz/` implementations while the refactor is in progress.
- The loader layer now understands both:
  - old flat component datasets like `DATASET_x`
  - new tracking output layout under `track/data/DATASET/components/x`
