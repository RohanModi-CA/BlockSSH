# GoHit

Sandbox for HIT-aware analysis work.

Current goal:

- preserve the existing `analysis/go` user surface
- prototype HIT-aware paths safely in a parallel tree
- keep one shared confirmed hit catalog per dataset
- support both `interhit` and optional `posthit` region construction

Current entry points:

```bash
python3 analysis/GoHit/FFT.py DATASET --hits
python3 analysis/GoHit/HitReview.py DATASET
python3 analysis/GoHit/Wavefunctions.py DATASET PEAKS
```

Notes:

- `FFT.py --hits` currently routes through the GoHit review GUI and saves a
  dataset-level catalog under `analysis/GoHit/out/hit_catalogs/`.
- non-HIT FFT behavior still delegates to the existing `analysis/go` path
- wavefunctions already understand the GoHit catalog surface, but the actual
  HIT-region wavefunction computation is not wired in yet
