# mothra-text

Pipeline and tools for HTR and HTR-OMR alignment on medieval chant manuscripts.
The primary artifact is `run_pipeline.py`, which runs a folio image through line segmentation,
optional Cantus text alignment, and word/syllable geometry generation. Two visualization tools
(Pipeline Inspector GUI and PAGE XML Viewer) let you inspect the output.

For comparative segmentation experiments and PyLaia baselines, see [`experiments/README.md`](experiments/README.md).

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
recto), `run_chain.py` logs a warning and resets the carry-over state rather than silently
passing stale words into the next folio's alignment:

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

Compare results with `scripts/compare_runs.py`. Load all output JSONs into the
Pipeline Inspector GUI for visual comparison.

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

See [`scripts/README.md`](scripts/README.md) for usage.

---

## Repo layout

```
mothra-text/
├── experiments/                    # comparative research (not part of main pipeline)
│   ├── README.md                   # experiments documentation
│   ├── run_htrflow.py              # YOLO/RTMDet segmentation runner
│   ├── run_all.py                  # runs all three models
│   ├── pipelines/                  # htrflow YAML configs for YOLO and RTMDet
│   └── pylaia_baseline/            # zero-shot PyLaia HTR baselines
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
│   ├── nw_chant_allocator.py
│   ├── syllable_segmentation.py
│   └── README.md
├── tests/                          # pytest suite (200+ tests)
├── page_viewer.py                  # PAGE XML Viewer desktop GUI
├── run_kraken.py                   # standalone Kraken BLLA runner + visualization
├── run_pipeline.py                 # end-to-end pipeline (single folio)
└── run_chain.py                    # automated multi-folio chaining wrapper
```

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

For experiment-specific dependencies (OpenMMLab stack for RTMDet, PyLaia environment),
see [`experiments/README.md`](experiments/README.md).

---

## Tests

```bash
conda activate line-seg-eval
python -m pytest tests/ -v
```
