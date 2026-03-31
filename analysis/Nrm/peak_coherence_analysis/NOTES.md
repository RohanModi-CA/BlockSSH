# Peak Coherence / Bicoherence Notes

## Goal

Classify the `0681ROT270X.csv` peaks as fundamental or non-fundamental using data-driven evidence rather than assumptions.

Primary working criteria:

- A fundamental line should recur stably across quiet regions and resist lower-frequency explanations.
- A non-fundamental line should show a reproducible incoming relation to lower-frequency parents.
- Outgoing child relations strengthen the case that a line is parent-like.

## Major scripts

- `run_peak_coherence_analysis.py`
  - Default-mode quiet-region segmentation.
  - Same-frequency bond coherence control.
  - Cross-frequency phase-lock tests.

- `run_peak_bicoherence_analysis.py`
  - Welch-style complex-window triad tests.
  - Main tool for `f1 + f2 -> f3`.

- `run_cross_dataset_triad_survey.py`
  - Compare `IMG_0681_rot270` and `IMG_0680_rot270`.

- `run_18_parent_comparison.py`
  - Direct comparison of `18` parent hypotheses.

- `run_0681_peak_taxonomy.py`
  - First raw-peak taxonomy over the CSV list.

- `run_0681_family_taxonomy.py`
  - More defensible family-level taxonomy above `3 Hz`.

- `run_0681_priority_report.py`
  - Explicit incoming-candidate checks for the strongest `>3 Hz` families.

## What has held up

- `18` is the strongest non-fundamental case so far.
  - In `0681`, `8.96 + 8.96 -> 18` is strongly above the null.
  - Nearby controls at `17.5` and `18.5` are not.
  - In `0680`, the same relation appears but broadens upward into an `18-ish` band.

- The strongest `0681` parent candidates remain around:
  - `3.37`
  - `3.97`
  - `6.36`
  - `8.91`

- `~16` has real evidence in `0681`, but it is weaker and less universal than the `18` case.

## What failed / what was misleading

- Same-frequency bond coherence is high for many lines and does not separate fundamentals from followers by itself.

- Raw per-peak taxonomy on the full CSV is too eager to invent local sum-rule explanations because the peak list is dense.

- The first family clustering pass merges some broad neighborhoods too aggressively, especially around `8-9 Hz` and `15-18 Hz`. It is useful scaffolding, but not final.

## Current best read

- Big peaks are the right place to triage first.
- A large peak is not automatically fundamental, but if a large peak resists lower-frequency explanations and also emits downstream children, that is strong parent-like evidence.
- A small peak with a strong incoming relation is much easier to call non-fundamental.

## Important corrections

- `16.60` cannot be treated as non-fundamental just because it has incoming coupling evidence.
- The first family-level taxonomy over-penalized incoming evidence and under-counted outgoing evidence for higher-frequency families.
- The calibrated taxonomy fixes that by:
  - rewarding amplitude
  - rewarding persistence
  - rewarding reproducibility in `0680`
  - rewarding outgoing pair evidence
  - only weakly penalizing incoming evidence

## Refined band split

Using a tighter family merge tolerance (`0.10 Hz`) gives a more sensible split above `3 Hz`, especially in the crowded `8-9 Hz` band:

- `8.557` -> tracked near `8.82`
- `8.742` and `8.950` -> tracked near `8.96`
- `15.823` and `16.025` no longer merge with `16.601`

This split is materially better than the original coarse family merge.

## Latest targeted results

- `8.962 + 8.962 -> 17.923` is strongly above null in `0681`.
- `8.820 + 8.820 -> 17.640` is not above null.
- `8.820 + 8.962 -> 17.782` is above null, but weaker than `8.962 + 8.962`.
- `7.956 + 8.106 -> 16.062` is strongly above null in `0681`.
- `16.60 + 6.37 -> 23.16` is strongly above null in `0681` and weakly above null in `0680 bond1`.
- `16.60 + 5.19 -> 21.96` is above null in `0680 bond1`.

These last two points matter: they show `16.60` also behaves like a parent, not just like a candidate child.

## Immediate next steps

- Split the broad `8-9 Hz` and `15-18 Hz` families using child-frequency scans rather than one-pass clustering.
- Re-run the family taxonomy after that split.
- Build a parent/child ranking over families:
  - incoming evidence
  - outgoing evidence
  - amplitude rank
  - cross-dataset reproducibility

## New work after the first calibrated taxonomy

- `run_parent_child_balance.py`
  - Simple balance plot from the calibrated taxonomy.
  - Useful as a quick visual, but still too dependent on the original family metrics.

- `run_family_edge_survey.py`
  - Cross-dataset survey over all family-level pair-sum edges.
  - Main advantage: it checks the same edge in `0681 bond0` and `0680 bond1`.
  - This is currently the best parent-vs-child graph.

- `run_key_peak_specificity.py`
  - Raw-peak triad checks with nearby fake controls.
  - Important because family representative frequencies can smear some exact raw-peak relations.

- `run_key_child_scans.py`
  - Frequency scans of `f3` for key parent pairs.
  - Main purpose: distinguish a narrow child line from a broad or shifted coupled band.

- `run_family_prominence.py`
  - Orthogonal test: local spectral prominence above neighboring frequencies.
  - This is useful because it does not rely on coupling at all.

- `run_family_synthesis_report.py`
  - Combines incoming/outgoing edge evidence with local prominence.
  - Current classes are only a working triage, not final truth.

## Stronger things that now seem true

- `8.96` is the cleanest parent-like high-frequency family so far.
  - Family `F15 ~ 8.962` has:
    - weak incoming edge score
    - strong outgoing edge score
    - high local prominence in both datasets
  - It is still the best current example of a line that behaves like a source rather than a follower.

- `18` is still the clearest non-fundamental family.
  - `8.95 + 8.95 -> f3` peaks near:
    - `17.95` in `0681`
    - `18.05` in `0680`
  - In `0681`, the exact raw peak at `18.361` is not the strongest point of the child band.
  - In `0680`, `18.361` is much more compatible with the same parent pair.
  - So the right object is the `18-ish` child band, not one exact frequency in all datasets.

- `15.94` and `16.05` still look follower-like.
  - `8.106 + 8.106 -> f3` peaks near `16.10-16.20` in both datasets.
  - `7.956 + 8.106 -> f3` also points into the same `16.0-16.2` band.
  - These families have:
    - strong incoming edge evidence
    - weak or negative outgoing evidence
    - modest local prominence

- `16.60` is not well described as a simple follower line.
  - The family-level incoming edge `7.712 + 8.96 -> 16.60` exists, but the best scanned child frequency is not locked sharply to `16.601`.
  - The best `f3` for that pair drifts:
    - `16.40` in `0681`
    - `16.05` in `0680`
  - That means the naive “incoming coupling implies child” interpretation is wrong here.
  - In the synthesis report, `16.60` is now only `mixed-coupled`, not `nonfundamental-leaning`.

## Important troubleshooting lessons

- Exact raw peaks and family representative frequencies are not interchangeable.
  - For some relations, the coupling localizes to a nearby band rather than the exact family center.

- Some older outgoing claims for `16.60` were too optimistic.
  - `16.601 + 6.369 -> 23.159` is above null in `0681`, but the scan peaks even more strongly nearer `22.8`.
  - `16.601 + 5.190 -> 21.960` does not hold up cleanly as an exact-target relation.
  - So `16.60` may still be parent-like, but those specific outgoing lines are not yet clean proofs.

- Local prominence is an important cross-check.
  - `F25 ~ 15.94` and `F26 ~ 16.05` have only modest prominence.
  - `F27 ~ 16.60` is also not a huge standout, but it is at least not weaker than the obvious follower pair around `16.0`.
  - `F15 ~ 8.96`, `F14 ~ 8.82`, `F07 ~ 6.36`, and `F01 ~ 3.37` are much stronger standalone lines by this metric.

## Current best triage

Most defensible parent-like families:

- `F01 ~ 3.37`
- `F02 ~ 3.97`
- `F07 ~ 6.36`
- `F14 ~ 8.82`
- `F15 ~ 8.96`
- `F17 ~ 11.31`
- `F18 ~ 12.05`

Most defensible non-fundamental-leaning families:

- `F19 ~ 12.61`
- `F20 ~ 13.05`
- `F21 ~ 13.21`
- `F25 ~ 15.94`
- `F26 ~ 16.05`

Mixed / still dangerous to over-interpret:

- `F27 ~ 16.60`
- `F28 ~ 18.36`
- `F23 ~ 14.24`

## Midband scan update

I added a direct `12.6-13.3 Hz` child-frequency scan:

- `4.766 + 7.956 -> f3`
  - peaks near `12.60` in `0681`
  - peaks near `12.80` in `0680`
  - the exact `12.607` target holds up well
  - this strengthens the case that `~12.61` is a real follower line

- `4.359 + 8.557 -> f3`
  - does **not** hold up in `0681`
  - does peak near `13.10` in `0680`

- `4.359 + 8.950 -> f3`
  - is weak in `0681`
  - peaks near the upper scan edge in `0680`

So the `12.61` family is now better supported than the `13.05-13.21` family pair.
Those `13`-region lines still have family-edge evidence, but the exact-target raw-peak scans are not yet clean enough to call them conclusive.

## Best next experiments from here

- Replace the one-shot family centers with scan-informed bands for:
  - `18-ish`
  - `16.0-16.2`
  - `16.6-16.9`
  - `22.8-23.2`

- Add a stricter “exact target vs broad band” score:
  - Is the coupling peaked at the listed line?
  - Or is the line just living inside a broader coupled band?

- Build a final peak-level report that maps each raw CSV peak onto:
  - a family
  - a band-specific coupling status
  - a prominence score
  - a provisional class
