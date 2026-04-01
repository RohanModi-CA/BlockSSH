Prototype workspace for nonlinear hit-strength analysis.

Current entry point:

```bash
python3 analysis/NL/prototype_hits.py IMG_0681_rot270
```

What it does right now:

- loads a dataset component and derives bond signals
- repairs obviously bad frame-time vectors using frame numbers
- builds a broadband sliding-spectrum energy trace
- detects candidate hits from that energy trace
- assigns per-hit strength metrics from an early post-hit window
- plots the aggregate time series, broadband energy with detections, and a sliding FFT

This is only steps 1 and 2 of the larger nonlinear-peak-scaling workflow.
