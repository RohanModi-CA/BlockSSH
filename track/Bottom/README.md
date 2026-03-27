# Bottom Tracking Workflow

This is the active user-facing tracking path for the refactor.

Run from the repo root:

```bash
python3 track/Bottom/0.VideoPrepareBottom.py DATASET
python3 track/Bottom/1.TrackRun.py DATASET
python3 track/Bottom/2.ProcessVerify.py DATASET
python3 track/Bottom/2b.ManualRepair.py DATASET
python3 track/Bottom/3.Label.py DATASET
```

Batch helpers:

```bash
python3 track/Bottom/B0.BatchPrepare.py
python3 track/Bottom/B1.BatchTrack.py
python3 track/Bottom/B2.BatchProcessVerify.py
```

Notes:

- `0.VideoPrepareBottom.py` stores transform and detection settings in
  `track/data/DATASET/params_bottom.json` when using the new flow.
- Existing datasets with only `params_black.json` are still accepted and can be
  migrated forward incrementally.
- `2.ProcessVerify.py` writes successful outputs under:
  `track/data/DATASET/components/x|y|a/track2_permanence.msgpack`
- If automatic repair is not sufficient, step 2 stops and tells you to run
  `2b.ManualRepair.py`.
- `3.Label.py` stores only site labels plus disabled sites. Bond information is
  expected to be derived downstream.
