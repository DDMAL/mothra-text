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

## Troubleshooting

### Too many lines are being detected

The pipeline is picking up music notation (neumes, staves) or non-chant text as lines.

- **Is text-region masking active?** — Masking is applied automatically when the upstream step succeeds. If you are seeing many false detections, the upstream step may not have run. Check whether masking was applied in the pipeline log.
- **Masking is active but neumes are still appearing?** — The masking expansion may be too large, revealing adjacent neume rows. Try lowering `--padding` to `10` (default is `15`). This is most common on manuscripts where text and music rows are closely packed.
- **Two-column folio detected as a single column?** — Use `--column-count 2` to force the correct layout.

---

### Lines are missing or cut short

The pipeline is not detecting all the chant text lines.

- **Is masking active?** — Masking is applied automatically by an upstream step. If it did not run (check the pipeline log for "Skipped silently"), Kraken sees the full image including notation and decorations, which can cause missed or merged lines. Contact your system administrator if the upstream step is consistently failing.
- **Masking is active but lines are still missing or cut short?** — The masking expansion may be too small to merge nearby word-level marks into full lines. Try raising `--padding` to `20` or `25`.
- **Still missing lines after adjusting padding?** — Try a custom segmentation model trained on similar material.

---

### Text alignment looks shifted or wrong *(Cantus mode only)*

The Cantus text is being matched to the wrong lines.

- **Check the folio identifier** — It must match the Cantus database exactly. Common mistakes: `5r` instead of `005r`, `p.100` instead of `100`. Fetch the Cantus CSV for your source and check which folio strings appear there.
- **First lines belong to the previous page's chant** — Use `--line-offset N` to skip the first N Cantus text entries before alignment begins. For example, if the first two lines of your folio finish a chant that started on the previous page, use `--line-offset 2`.
- **Chant continues from a previous folio run** — Supply `--prev-folio-state` with the state JSON written by the previous folio's run. This carries over any unfinished chant text.

---

### Column detection is wrong

- **Two columns detected as one** — Use `--column-count 2`.
- **One column detected as two** — Use `--column-count 1`.
- **Auto-detect is almost right but occasionally wrong** — Adjust `--column-bimodal-threshold`. Lower values make the detector require a deeper gap between columns (fewer false two-column detections); higher values accept a shallower gap (fewer false one-column detections). Default is `0.5`.

---

### I only need line/word positions, not the recognised text

Use `--stub-mode`. The pipeline skips text recognition entirely but still produces full word and syllable geometry. Useful when no recognition model is available or when OCR quality doesn't matter.

---

## Advanced options reference

These options are available on the command line. In the GUI they appear in the Advanced panel.

**Text-region masking (automatic)** — An upstream step automatically supplies a file marking which regions of the folio image are chant text. The pipeline blacks out everything outside those regions before running line detection, which is the single most effective way to reduce false detections from neumes, staves, and decorations. If the upstream step does not return a result, masking is skipped silently and the pipeline runs on the full image.

**`--padding PX`** *(default: 15)* — When masking is enabled, each text region is expanded outward by this many pixels before the mask is applied. A larger value helps merge nearby word-level detections into continuous line strips, which Kraken needs to find a complete line. Too large and the mask bleeds into adjacent music rows, causing false detections. Reduce to around 10 on manuscripts where text and neume rows are tightly packed; increase toward 20–25 on manuscripts with more spacing between words.

**`--stub-mode`** — Skip text recognition. All lines get empty text; the pipeline still runs and produces word/syllable geometry. Takes precedence over `--recognition-model`.

**`--device cpu|mps|cuda`** *(default: cpu)* — Which processor to use for Kraken inference. `cpu` works on any machine. `mps` uses Apple Silicon's GPU and runs noticeably faster on a Mac. `cuda` uses an NVIDIA GPU. Only change this if you have a compatible GPU and want faster processing.

**`--line-offset N`** *(default: 0)* — Skip the first N entries in the Cantus text before starting alignment. Use this when the top of your folio image continues a chant that began on the previous page — those first lines belong to the old chant and should not be matched to the new one.

**`--column-bimodal-threshold FLOAT`** *(default: 0.5)* — Controls how pronounced the gap between two columns must be for auto-detection to declare the folio two-column. Lower values require a more obvious gap (safer, fewer false positives). Higher values accept a subtler gap (useful for manuscripts where the two columns are closer together). If auto-detection is wrong, it is simpler to use `--column-count` instead.

**`--prev-folio-state PATH`** — Provide a state file written by the previous folio's run (via `--folio-state-out`). This passes any chant text that spilled over from the previous page into the current alignment. Only needed when processing consecutive folios in separate runs; the automated `run_chain.py` script handles this automatically.
