# mothra-text

Pipeline and tools for HTR and HTR-OMR alignment on medieval chant manuscripts.
The primary artifact is `run_pipeline.py`, which runs a folio image through line segmentation,
optional Cantus text alignment, and word/syllable geometry generation. Two visualization tools
(Pipeline Inspector GUI and PAGE XML Viewer) let you inspect the output.

For comparative segmentation experiments, see [`experiments/README.md`](experiments/README.md).

---

## Projects

### 1. End-to-end PoC pipeline

`run_pipeline.py` runs a single folio image through the full pipeline:

1. **Kraken BLLA** — baseline line segmentation
2. **Column clustering** — auto-detect 1 vs 2 columns; sort lines into reading order
3. **Co-linear segment fusion** — fuse BLLA sub-segments belonging to the same physical
   text line (≥50% y-extent overlap) into logical lines, correcting BLLA over-segmentation
   on chant manuscripts with neume notation
4. **Kraken HTR** — text recognition per logical line
5. **NW chant allocator** *(skipped in OCR-only mode)* — align Cantus CSV text to detected
   lines via Needleman-Wunsch, using volpiano break markers as alignment anchors; supports
   folio-to-folio continuation via JSON sidecar (`--folio-state-out`); in no-volpiano mode,
   automatically locates where this folio's first chant begins via NW matching and assigns
   pre-start lines to the previous folio's continuation (see `locate_folio_start` and
   `pre_start_suffix_align` in [`steps/README.md`](steps/README.md))
6. **Word segmentation** — distribute ground-truth Cantus words across each line's pixel
   extent; falls back to OCR word splitting when no Cantus text is available (OCR-only mode
   or lines with no match)
7. **Syllable segmentation** — subdivide each word node into character-proportional
   syllable regions using Latin syllabification from `volpiano-display-utilities`

See [`steps/README.md`](steps/README.md) for details on each step.

```bash
python run_pipeline.py \
    --image path/to/folio.jpg \
    --folio "006r" \
    --source-id 123672 \
    --export-json ~/Downloads/DDMAL/006r.json
```

**Key flags:**

| Flag | Description |
|---|---|
| `--folio STR` | Folio identifier (e.g. `"006r"`). Required when `--csv` or `--source-id` is given; defaults to the image filename stem in OCR-only mode. |
| `--source-id INT` | Cantus source ID (fetched from cantusdatabase.org). Omit with `--csv` to enter OCR-only mode. |
| `--csv PATH` | Local Cantus-format CSV file. Omit with `--source-id` to enter OCR-only mode. |
| `--segmentation-model PATH` | Custom Kraken BLLA model (`.mlmodel` or `.safetensors`); omit for Kraken built-in |
| `--column-count {1,2}` | Declare column count; skips bimodal auto-detection |
| `--recognition-model PATH` | Kraken HTR model; defaults to Tridis if installed |
| `--stub-mode` | Skip text recognition; pipeline still runs using ground-truth text |
| `--prev-folio-state PATH` | JSON sidecar from the previous folio run (post-77 continuation words; Cantus mode only) |
| `--folio-state-out PATH` | Write folio state JSON for the next folio run (Cantus mode only) |
| `--export-json PATH` | Write output JSON for the Pipeline Inspector GUI |
| `--mei-json PATH` | Write MEI Text Alignment JSON to an explicit path (overrides `--output-dir`) |
| `--output-dir PATH` | Directory for auto-named MEI JSON output. Requires `--source-id`/`--csv` and `--folio`. Output is named `{RISM-code}_{shelfmark}_{folio}.json` (e.g. `CH-E_611_001r.json`). The `"folio"` field inside the JSON also uses this regularized name. |
| `--no-skip-misdetected-lines` | Allocate Cantus text to every detected line, including boxes far too small to hold it whose OCR read nothing. By default such boxes are flagged and skipped (see 1d) |
| `--misdetect-width-ratio FLOAT` | Maximum box width, as a fraction of the page's typical text-line width, for a line to be eligible to be skipped as a misdetection (default 0.35) |
| `--no-drop-offarea-boxes` | Allocate Cantus text to every detected line, including boxes lying almost entirely outside the main chant text area (folio numbers, running heads, marginal notes). By default such boxes are dropped before fusion (see 1e) |
| `--area-keep-threshold FLOAT` | Minimum fraction of a line's own area that must overlap the main chant text area for the line to be kept (default 0.50) |
| `--debug-ocr` | Print per-line OCR transcripts and NW alignment detail; in OCR-only mode also prints a startup banner and lists any ignored flags |

**OCR-only mode:** When neither `--csv` nor `--source-id` is given, the pipeline skips
Cantus data loading and NW alignment entirely. Steps 1–4 run normally; word boundaries come
from OCR word splitting and syllables are Latin-syllabified from the OCR text. The exported
JSON will contain `"mode": "ocr_only"` instead of `"cantus_aligned"`. Flags
`--prev-folio-state` and `--folio-state-out` are ignored with a warning.

```bash
# OCR-only (no Cantus data needed)
python run_pipeline.py \
    --image path/to/folio.jpg \
    --export-json ~/Downloads/DDMAL/folio.json
```

**Recognition model:** The Tridis model (`Tridis_Medieval_EarlyModern.mlmodel`) is used by
default if installed via htrmopo. To install:
```bash
python -m htrmopo get 10.5281/zenodo.10788591
```
If no model is found and `--stub-mode` is not given, the pipeline exits with an error.
Use `--stub-mode` to skip recognition entirely (pipeline still produces GT word/syllable geometry).

**Multi-folio runs (manual):**
```bash
# First folio
python run_pipeline.py --image 006r.jpg --folio 006r --source-id 123672 \
    --export-json ~/Downloads/DDMAL/006r.json --folio-state-out state_006r.json

# Next folio, with continuation from the previous
python run_pipeline.py --image 007v.jpg --folio 007v --source-id 123672 \
    --prev-folio-state state_006r.json --export-json ~/Downloads/DDMAL/007v.json
```

**Multi-folio runs (automated):** Use `run_chain.py` to chain any number of consecutive
folios in a single command — intermediate `FolioState` sidecar files are managed
automatically. If the provided folios are not actually consecutive pages (recto→verso→next
recto), `run_chain.py` logs a warning and resets the carry-over state, then falls back to
`build_flat_text_and_anchors`' own CSV-scanning fallback (see `steps/README.md`) for that
folio — the same one a standalone run of it would use:

```bash
# Auto-named MEI JSON outputs (recommended for batch use)
python run_chain.py \
    --images 006r.jpg 007v.jpg 008r.jpg \
    --folios 006r 007v 008r \
    --source-id 123672 \
    --output-dir ~/Downloads/DDMAL/
# Produces: CH-E_611_006r.json, CH-E_611_007v.json, CH-E_611_008r.json

# Explicit output paths (legacy / one-off)
python run_chain.py \
    --images 006r.jpg 007v.jpg 008r.jpg \
    --folios 006r 007v 008r \
    --source-id 123672 \
    --mei-json ~/Downloads/DDMAL/006r.json \
               ~/Downloads/DDMAL/007v.json \
               ~/Downloads/DDMAL/008r.json
```

| Flag | Description |
|---|---|
| `--images PATH [...]` | Ordered folio image paths |
| `--folios STR [...]` | Folio identifiers matching the CSV (same order as `--images`) |
| `--output-dir PATH` | Directory for auto-named MEI JSON outputs (`{RISM-code}_{shelfmark}_{folio}.json` per folio). Recommended for batch use. |
| `--mei-json PATH [...]` | One explicit MEI JSON path per folio; takes precedence over `--output-dir` |
| `--export-json PATH [...]` | One pipeline inspector JSON path per folio; parent dirs created automatically |
| `--folio-states-dir PATH` | Save intermediate `state_{folio}.json` files here for debugging |
| `--debug-ocr` | Print per-line OCR and NW alignment detail for every folio |
| `--mothra-jsons-dir PATH` | Directory containing mothra annotation JSONs named `{image_stem}.json`, one per folio (produced by `scripts/run_mothra_inference.py --out-dir`). Masks each folio's image before segmentation; a missing per-folio JSON logs a warning and runs that folio unmasked. |
| `--padding PX` | Pixels added around each text bbox before masking (default 15). Only used when `--mothra-jsons-dir` is given. |
| `--skip-masking` | Skip text-region masking even if `--mothra-jsons-dir` is given. |
| `--no-skip-misdetected-lines` | Allocate Cantus text to every detected line, including boxes far too small to hold it whose OCR read nothing (see 1d). |
| `--misdetect-width-ratio FLOAT` | Maximum box width, as a fraction of the page's typical text-line width, for a line to be eligible to be skipped as a misdetection (default 0.35). |
| `--no-drop-offarea-boxes` | Allocate Cantus text to every detected line, including boxes lying almost entirely outside the main chant text area (see 1e). |
| `--area-keep-threshold FLOAT` | Minimum fraction of a line's own area that must overlap the main chant text area for the line to be kept (default 0.50). |

All model and device flags from `run_pipeline.py` (`--segmentation-model`,
`--recognition-model`, `--device`, `--stub-mode`, `--column-count`,
`--column-bimodal-threshold`) are forwarded unchanged to every folio run.
The chain aborts on the first failure to avoid propagating corrupt state.

The manual `--prev-folio-state` / `--folio-state-out` approach above remains
available for one-off runs or non-consecutive folios.

---

### 1b. Text-region masking

In the production pipeline, a mothra text-detection JSON is supplied automatically
by an upstream step and passed to `--mothra-json`; the pipeline skips masking silently
if the upstream step does not return a result. For local research runs, pass
`--mothra-json` directly to black out non-text regions (staves, neumes, decorations)
before Kraken BLLA runs.

Use `--padding` (default 15 px) to control how much each text bbox is expanded to help
Kraken form full lines. Reduce to ~10 px on manuscripts where text and neume rows
are closely packed.

```bash
python run_pipeline.py \
    --image path/to/folio.jpg \
    --folio "012v" \
    --source-id 599679 \
    --mothra-json path/to/folio.json \
    --export-json ~/Downloads/DDMAL/mothra_masked_12v.json
```

**Masking flags:**

| Flag | Description |
|---|---|
| `--mothra-json PATH` | Mothra annotation JSON for this folio. Blacks out non-text regions before line segmentation. Omit to run without masking. |
| `--padding PX` | Pixels added around each text bbox before masking (default 15). |
| `--skip-masking` | Skip text-region masking even if `--mothra-json` is given. |

**Programmatic usage:** masking is also available when calling `run()` directly
as a library:

```python
from run_pipeline import run
collection, manifest = run(
    image_path="path/to/folio.jpg",
    folio="012v",
    source_id=599679,
    mothra_json_path="path/to/folio.json",
    padding=15,
)
```

Pass `mothra_json_path=None` (the default) to skip masking. Callers that do not
pass this argument — including `run_chain.py` — are unaffected.

### 1c. Pre-NW music region filter

When the production pipeline supplies music-region bounding boxes (the YOLO-detected
stave/neume areas), `run()` can drop any BLLA-detected line that substantially overlaps
a music region **before** Stage 4 (NW chant allocation). This prevents spurious BLLA
baselines near music staves from consuming GT word slots and shifting all subsequent
lines off by one.

This filter is not exposed as a CLI flag — it is only used by the mothra API layer
(`text-service/main.py`), which passes the music boxes from the YOLO annotation. CLI
runs and `run_chain.py` pass `music_boxes=None` (the default) and are unaffected.

**Programmatic usage:**

```python
from run_pipeline import run
collection, manifest = run(
    image_path="path/to/folio.jpg",
    folio="012v",
    source_id=599679,
    music_boxes=[[x0, y0, x1, y1], ...],   # absolute pixel coords
    music_overlap_threshold=0.30,           # optional, default 0.30
)
# Lines dropped before NW are on collection._music_filter_dropped
```

Compare results with `scripts/compare_runs.py`. Load all output JSONs into the
Pipeline Inspector GUI for visual comparison.

**How this repo is invoked in production:** the `--mothra-json`/`mothra_json_path` and
`music_boxes` inputs described in 1b/1c above are exactly what the [`mothra`](https://github.com/DDMAL/mothra)
landing-page repo's `text-service/main.py` passes to this repo's `run()` over an internal HTTP
call — `mothra-text` is included there as a git submodule. See
[DEEP_DIVE.md §9a](DEEP_DIVE.md#9a-relationship-to-the-mothra-repo-landing-page--text-service)
for a summary of that integration, and [DDMAL/mothra#151](https://github.com/DDMAL/mothra/issues/151)
for the full architecture write-up.

---

### 1d. Misdetected line skipping

BLLA sometimes draws a small box around a non-chant area — a neume group, a clef sliver, an
initial. The OCR model correctly reads nothing there, but the NW allocator cannot tell "no
text because this isn't text" from "no text because OCR failed", so it hands the box a whole
line's worth of Cantus words and every following line shifts.

`allocate_lines` now flags and skips such a box: it is assigned no text and the pointer stays
put, so its words go to the next box in the existing reading order. A box is only skipped when
all four signals agree — OCR read essentially nothing, the box is far narrower than the page's
typical text line, it cannot physically hold nearly as many characters as it is being offered,
and at least one word was actually being offered. See
[`steps/README.md`](steps/README.md#allocate_linesflat_text-sorted_labels-ocr_texts-column_count1-)
for the thresholds and the page statistics they are measured against.

Each skip is logged twice: once as a `misdetected_line_skipped` validation flag with the full
reasoning, and once as a summary line naming every skipped box and its bbox:

```
WARNING  Validation flag [misdetected_line_skipped]: Line fused_10: bbox [1019,1966,1079,2073]
         is 60px wide (4% of the page's typical line) and OCR read no text, but allocation
         wanted 6 word(s) (48 chars) starting 'seculorum' — the box holds ~2 char(s).
         Treating as a non-text detection: skipped, words left for the next line.
WARNING    Skipped 1 misdetected (non-text) line(s): fused_10 [1019,1966,1079,2073] = 002v_region10
```

The skipped box is **kept** in the output — it still appears in `--export-json` and the GUI,
just with no words — so a skip can be checked by eye. Empty syllables are already filtered out
of the MEI JSON, so that output is unaffected.

Unlike the music-region filter in 1c this needs no external annotation: it works from the
page's own line geometry, so it protects CLI and `run_chain.py` runs too. The two are
complementary rather than alternatives — when YOLO music boxes *are* available, 1c removes
part of this class of box earlier and on stronger evidence, but it is scoped to music regions
and cannot see a folio number or a marginal mark. Pass `--no-skip-misdetected-lines` to
disable, or lower `--misdetect-width-ratio` to skip fewer boxes.

**Masking reduces these boxes but does not eliminate them, so this rule applies to masked
runs too.** Text-region masking (1b) attacks the problem at the source — BLLA never sees the
non-text pixels, so it cannot draw a box there — and where it works it is the better fix. On
CH-Fco Ms. 2 002r an unmasked run produces 15 fused lines including two non-text boxes (a
pencil folio number and a single neume) that this rule skips, while the same folio with
`--mothra-json` produces 13 fused lines, no non-text boxes at all, and leaves this rule fully
inert. But masking is itself a detection step and leaks: the NZ-Wt MSR-03 002v example was
produced *with* masking and still contained a 21 px non-text box that consumed 6 words and
shifted the rest of the page. Keep this rule on for masked and unmasked runs alike.

Not applicable in OCR-only mode: without Cantus text there is no shared text pointer for a
non-text box to corrupt.

---

### 1e. Main text area filtering

BLLA also draws boxes on marginalia that sit clearly outside the chant text block itself —
folio numbers, running heads, left-margin scribal notes (e.g. a psalm/canticle incipit cue
like "david"). Unlike 1d, this class of box is not offered too few characters to hold — its
OCR often reads fine — so it isn't caught by the misdetect rule at all, and by the time
`fuse_colinear_segments` runs, a box that vertically overlaps a real line has already been
merged into it, hiding its marginal position from every later stage.

Before fusion, the pipeline computes the main chant text area per column from the page's own
line geometry: boxes are grouped into rows by y-overlap (the same rule `fuse_colinear_segments`
itself uses), and a row's *combined* width — not any single box's width — decides whether it
anchors the text block, so a real line that BLLA fragmented into several small boxes is still
recognized as real. Any box lying almost entirely outside the resulting area is dropped before
fusion, on `collection._offarea_filter_dropped`.

```
INFO  folio 009v: pre-NW drop — line bbox [492,1705,633,1799] lies outside the main text area (overlap=0.00)
```

Like 1d this needs no external annotation and protects CLI and `run_chain.py` runs too. Pass
`--no-drop-offarea-boxes` to disable, or raise `--area-keep-threshold` to drop fewer boxes.

**Scope:** this only catches boxes outside the block's true x/y footprint. Marginalia that
sits *within* the block — an interlinear rubric cue between two real lines, for instance —
looks geometrically identical to a genuine short line and is deliberately left alone; telling
those apart needs region classification, not geometry, and is out of scope here.

---

### 2. Pipeline Inspector GUI

A browser-based viewer for inspecting pipeline output — folio image overlaid with
line polygons, word bounding boxes, and syllable regions, with per-layer toggles.

**Live:** https://ddmal.github.io/mothra-text/ — load any folio image + pipeline JSON
generated by `run_pipeline.py --export-json`, no install required.

**Word box colors:** teal = Cantus ground truth, rose = OCR fallback (no GT available).

See [`gui/README.md`](gui/README.md) for usage and local development instructions.

---

### 3. PAGE XML Viewer

A lightweight Python desktop viewer for inspecting PAGE XML annotation files overlaid on
their source manuscript images. Useful for verifying ground-truth annotations produced by
`scripts/mothra_to_page.py` or any other PAGE XML source without needing a browser.

**Launch:**
```bash
python page_viewer.py                          # open files via dialog
python page_viewer.py annotation.xml           # load XML, locate image interactively
python page_viewer.py image.jpg annotation.xml # pre-load both on startup
```

**Features:**
- Renders TextRegions, TextLines, Words, Baselines, and Glyphs as colour-coded overlays
- Per-layer visibility toggles
- Scroll-wheel zoom centred on cursor; click-and-drag pan
- Click an annotation to see its ID, type, text, and attributes in the sidebar
- No extra dependencies beyond **Pillow** (already required by other scripts)

---

### 4. Scripts

Utility and conversion scripts in `scripts/`:

| Script | Description |
|---|---|
| `mothra_to_page.py` | Convert Mothra Annotator JSON → PAGE XML (for BLLA training data) |
| `convert_to_mei_input.py` | Convert pipeline JSON → MEI Text Alignment JSON |
| `debug_column_detection.py` | Visualize bimodal column detection coverage profile |
| `run_mothra_inference.py` | Run YOLOv11 mothra models over folio images → mothra annotation JSON |
| `compare_runs.py` | Compare pipeline output JSONs across different approaches/runs |
| `visualize_mothra.py` | Overlay mothra annotation bboxes on a folio image |

See [`scripts/README.md`](scripts/README.md) for usage.

---

## Repo layout

```
mothra-text/
├── experiments/                    # comparative research (not part of main pipeline)
│   ├── README.md                   # experiments documentation
│   ├── run_htrflow.py              # YOLO/RTMDet segmentation runner
│   ├── run_all.py                  # runs all three models
│   └── pipelines/                  # htrflow YAML configs for YOLO and RTMDet
├── gui/                            # Pipeline Inspector browser app
│   └── README.md
├── scripts/                        # utility and conversion scripts
│   └── README.md
├── steps/                          # pipeline step implementations
│   ├── column_clustering.py
│   ├── ground_truth_word_segmentation.py
│   ├── gt_manifest.py
│   ├── kraken_recognition.py
│   ├── kraken_segmentation.py
│   ├── mothra_mask.py
│   ├── nw_chant_allocator.py
│   ├── syllable_segmentation.py
│   └── README.md
├── docs/                           # user-facing documentation
│   ├── user_guide.md
│   └── user_decision_tree.md
├── tests/                          # pytest suite (200+ tests)
├── page_viewer.py                  # PAGE XML Viewer desktop GUI
├── run_kraken.py                   # standalone Kraken BLLA runner + visualization
├── run_pipeline.py                 # end-to-end pipeline (single folio)
└── run_chain.py                    # automated multi-folio chaining wrapper
```

---

## Documentation

| Doc | Covers |
|---|---|
| [`DEEP_DIVE.md`](DEEP_DIVE.md) | Full architecture deep dive: every pipeline stage, key data structures, known limitations, and pitfalls/gotchas |
| [`steps/README.md`](steps/README.md) | Per-module reference for `steps/` |
| [`docs/user_guide.md`](docs/user_guide.md) | Troubleshooting and CLI option reference for end users |
| [`docs/user_decision_tree.md`](docs/user_decision_tree.md) | GUI flag mapping and a decision tree for choosing pipeline options |
| [`gui/README.md`](gui/README.md) | Pipeline Inspector GUI usage and local development |
| [`scripts/README.md`](scripts/README.md) | Utility/conversion script reference |
| [`experiments/README.md`](experiments/README.md) | Comparative segmentation research (not part of the main pipeline) |
| [DDMAL/mothra#151](https://github.com/DDMAL/mothra/issues/151) (external) | Full architecture write-up of how the `mothra` landing-page repo integrates this repo in production — see also [DEEP_DIVE.md §9a](DEEP_DIVE.md#9a-relationship-to-the-mothra-repo-landing-page--text-service) |

---

## Data

Folio images and model outputs are stored on HuggingFace, not in this repo.
Pull them locally before running:

```bash
# Pull folio images → data/folios/
ddmal-hfsync pull-groundtruth --shared --dir data

# Pull model outputs → outputs/
ddmal-hfsync pull-runs --project mothra-text --model kraken --dir outputs/kraken_blla
```

See [DDMAL/ddmal_hfsync](https://github.com/DDMAL/ddmal_hfsync) for setup instructions
(`~/.hfconfig` must be configured).

---

## Environment setup

```bash
conda create -n line-seg-eval python=3.10 -y
conda activate line-seg-eval
pip install -r requirements.txt
```

All dependencies (including transitive) are pinned in `requirements.txt`. To update
after adding a new package, re-run `pip freeze > requirements.txt` in the active conda
env and commit the result.

For experiment-specific dependencies (OpenMMLab stack for RTMDet),
see [`experiments/README.md`](experiments/README.md).

---

## Tests

```bash
conda activate line-seg-eval
python -m pytest tests/ -v
```
