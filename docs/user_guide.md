# Mothra-Text User Guide

This guide helps you get good results from the mothra-text pipeline and diagnose common problems. For a complete map of every option, see [`user_decision_tree.md`](user_decision_tree.md).

---

## What does the pipeline do?

The pipeline takes a scanned folio image and returns the positions of every text line, word, and syllable on the page. If you supply a Cantus source ID, words come from the Cantus database of medieval chant — so the text is ground-truth rather than computer-read. Without Cantus data the pipeline still works, using its own text recognition to split lines into words.

---

## Quick-start checklist

Minimum inputs for a successful run:

1. **Folio image** — JPEG, PNG, or TIFF
2. *(For Cantus-aligned mode)* **Cantus source ID** — the number from the source detail page on cantusdatabase.org
3. *(For Cantus-aligned mode)* **Folio identifier** — exactly as it appears in the Cantus database (e.g. `005r`, not `5r`)

Text-region masking is applied automatically by an upstream step — no extra input is needed from you.

---

## Outputs

The pipeline produces two optional output files:

**MEI JSON** (`--output-dir PATH` or `--mei-json PATH`) — The primary output consumed by downstream steps such as mothra-encoding. In Cantus-aligned mode, the file is automatically named using information from the Cantus database:

```
{RISM-code}_{shelfmark}_{folio}.json
e.g.  CH-E_611_001r.json
```

The RISM code and shelfmark are extracted from the Cantus CSV the pipeline already downloads — no extra input is needed. The `"folio"` field inside the JSON uses the same regularized identifier.

**Pipeline Inspector JSON** (`--export-json PATH`) — An optional secondary output for visual inspection in the Pipeline Inspector GUI. Written only when an explicit path is given; not needed for downstream processing.

---

## Optional inputs

### Column count

The pipeline auto-detects whether a folio has one or two columns. If auto-detection is wrong, override it with `--column-count 1` or `--column-count 2`. When in doubt, `--column-count 2` is the safer override on two-column folios — it still locates the split position automatically.

### Segmentation model

By default the pipeline uses Kraken's built-in BLLA model for line segmentation. If a fine-tuned model trained on similar manuscript material is available, pass it via `--segmentation-model` (`.mlmodel` or `.safetensors` format). When using a model optimised for a specific column layout, also set `--column-count`.

### Recognition / OCR model

By default the pipeline uses the Tridis Medieval/EarlyModern model if installed via htrmopo (`python -m htrmopo get 10.5281/zenodo.10788591`). To use a different HTR model, pass it via `--recognition-model`. If no model is found and `--stub-mode` is not given, the pipeline exits with an error. To skip recognition entirely (produce word/syllable geometry without text), use `--stub-mode`.

---

## Troubleshooting

### Too many lines are being detected

The pipeline is picking up music notation (neumes, staves) or non-chant text as lines.

- **Is text-region masking active?** — Masking is applied automatically when the upstream step succeeds. If you are seeing many false detections, the upstream step may not have run. Check whether masking was applied in the pipeline log.
- **Masking is active but neumes are still appearing?** — The masking expansion may be too large, revealing adjacent neume rows. Try lowering `--padding` to `10` (default is `15`). This is most common on manuscripts where text and music rows are closely packed.
- **Try a fine-tuned segmentation model** — a model trained on similar manuscript material will generally segment more accurately than the default Kraken BLLA and produce fewer false detections.
- **Two-column folio detected as a single column?** — Use `--column-count 2` to force the correct layout.

---

### Lines are missing or cut short

The pipeline is not detecting all the chant text lines.

- **First: try a fine-tuned segmentation model** — a model trained on similar manuscript material is the most reliable fix for systematically missed lines.
- **If masking is active and lines are being cut short** — the masking expansion may be too small to merge nearby word-level marks into full line strips. Try raising `--padding` to `20` or `25`.
- **If masking seems to be removing too much** — try disabling masking via `--skip-masking` (or omit `--mothra-json` in CLI mode) to confirm whether masking is the cause.
- **Still missing lines?** — Try a different segmentation model or check whether the folio image quality is sufficient for Kraken to detect lines.

---

### Text alignment looks shifted or wrong *(Cantus mode only)*

The Cantus text is being matched to the wrong lines.

- **Check the folio identifier** — It must match the Cantus database exactly. Common mistakes: `5r` instead of `005r`, `p.100` instead of `100`. Fetch the Cantus CSV for your source and check which folio strings appear there.
- **First lines belong to the previous page's chant** — The pipeline handles this automatically by scanning the Cantus CSV for the preceding folio's continuation words. If the automatic inference is wrong, supply `--prev-folio-state` with the state JSON from the previous folio's run to provide the continuation explicitly. For consecutive folio batches, `run_chain.py` manages this automatically.
- **Chant continues from a previous folio run** — Supply `--prev-folio-state` with the state JSON written by the previous folio's run. This carries over any unfinished chant text. For consecutive folio batches, `run_chain.py` handles this automatically.
- **OCR recognition quality is poor** — if Cantus alignment is failing because OCR text doesn't match the expected chant, try a fine-tuned recognition model via `--recognition-model`.

---

### Column detection is wrong

- **Two columns detected as one** — Use `--column-count 2`.
- **One column detected as two** — Use `--column-count 1`.
- **Auto-detect is almost right but occasionally wrong** — Adjust `--column-bimodal-threshold`. Lower values make the detector require a deeper gap between columns (fewer false two-column detections); higher values accept a shallower gap (fewer false one-column detections). Default is `0.5`.

---

## Advanced options reference

For a quick-reference table of all flags and their GUI placements, see [`user_decision_tree.md`](user_decision_tree.md).

These options are available on the command line. In the GUI they appear in the Advanced panel.

**`--skip-masking`** — When present, bypasses text-region masking even if the upstream step has supplied a mothra JSON. Masking is also skipped automatically when the upstream step does not return a result. Use this when masking is actively causing problems — for example, if too many lines are being suppressed — and you want to run on the raw image. In CLI mode, omitting `--mothra-json` has the same effect.

**`--padding PX`** *(default: 15)* — When masking is enabled, each text region is expanded outward by this many pixels before the mask is applied. A larger value helps merge nearby word-level detections into continuous line strips, which Kraken needs to find a complete line. Too large and the mask bleeds into adjacent music rows, causing false detections. Reduce to around 10 on manuscripts where text and neume rows are tightly packed; increase toward 20–25 on manuscripts with more spacing between words.

**`--stub-mode`** — Skip text recognition. All lines get empty text; the pipeline still runs and produces word/syllable geometry. Takes precedence over `--recognition-model`.

**`--device cpu|mps|cuda`** *(default: cpu)* — Which processor to use for Kraken inference. `cpu` works on any machine. `mps` uses Apple Silicon's GPU and runs noticeably faster on a Mac. `cuda` uses an NVIDIA GPU. Only change this if you have a compatible GPU and want faster processing.

**`--column-bimodal-threshold FLOAT`** *(default: 0.5)* — Controls how pronounced the gap between two columns must be for auto-detection to declare the folio two-column. Lower values require a more obvious gap (safer, fewer false positives). Higher values accept a subtler gap (useful for manuscripts where the two columns are closer together). If auto-detection is wrong, it is simpler to use `--column-count` instead.

**`--prev-folio-state PATH`** — Provide a state file written by the previous folio's run (via `--folio-state-out`). This passes any chant text that spilled over from the previous page into the current alignment. Only needed when processing consecutive folios in separate runs; the automated `run_chain.py` script handles this automatically.
