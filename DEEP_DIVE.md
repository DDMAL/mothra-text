# mothra-text Deep Dive Report

**Repo:** `/Users/cassiebastress/Documents/DDMAL/line-seg-eval`
**Date written:** 2026-06-18  
**Last updated:** 2026-07-13

---

## Table of Contents

1. [Purpose and Big Picture](#1-purpose-and-big-picture)
2. [Pipeline Stages](#2-pipeline-stages)
   - [Stage 0 — Mothra Text-Region Masking](#stage-0--mothra-text-region-masking-inside-run)
   - [Stage 1 — Kraken BLLA Line Segmentation](#stage-1--kraken-blla-line-segmentation-krakensegmentation)
   - [Stage 2 — Column Clustering and Co-linear Fusion](#stage-2--column-clustering-and-co-linear-fusion-cluster_columns--fuse_colinear_segments)
   - [Stage 3 — Kraken HTR Text Recognition](#stage-3--kraken-htr-text-recognition-krakenrecognition)
   - [Stage 4 — NW Chant Allocator or OCR-Only Mode](#stage-4--nw-chant-allocator-allocate_lines-or-ocr-only-mode)
   - [Stage 5 — GT Word Segmentation](#stage-5--gt-word-segmentation-groundtruthwordsegmentation)
   - [Stage 6 — Syllable Segmentation](#stage-6--syllable-segmentation-syllablesegmentation)
3. [File Structure](#3-file-structure)
4. [Framework Connections](#4-framework-connections)
   - [HTRflow](#htrflow)
   - [Kraken](#kraken)
   - [Bio.Align (Biopython)](#bioalign-biopython)
   - [volpiano-display-utilities](#volpiano-display-utilities)
   - [platformdirs](#platformdirs)
   - [htrmopo](#htrmopo)
5. [NW Chant Allocator — Deep Dive](#5-nw-chant-allocator--deep-dive)
   - [5a. build_flat_text_and_anchors()](#5a-build_flat_text_and_anchors)
   - [5b. allocate_lines() — the NW allocation loop](#5b-allocate_lines--the-nw-allocation-loop)
   - [5c. _defuse_manifest()](#5c-_defuse_manifest)
   - [5d. Folio state persistence](#5d-folio-state-persistence)
6. [Key Data Structures](#6-key-data-structures)
   - [HTRflow tree hierarchy](#htrflow-tree-hierarchy)
   - [FlatTextData](#flattextdata)
   - [Pipeline JSON output schema](#pipeline-json-output-schema)
   - [Mothra annotation JSON schema](#mothra-annotation-json-schema)
7. [GUI Architecture](#7-gui-architecture)
8. [Multi-Folio Chaining (run_chain.py)](#8-multi-folio-chaining-run_chainpy)
9. [Mothra Text-Detection Integration](#9-mothra-text-detection-integration)
   - [MothraImageMask](#motherimagemask-stepsmothra_maskpy)
   - [scripts/run_mothra_inference.py](#scriptsrun_mothra_inferencepy)
   - [scripts/visualize_mothra.py](#scriptsvisualize_mothrapy)
   - [scripts/compare_runs.py](#scriptscompare_runspy)
   - [9a. Relationship to the mothra Repo (Landing Page / text-service)](#9a-relationship-to-the-mothra-repo-landing-page--text-service)
10. [Tools and External Dependencies](#10-tools-and-external-dependencies)
11. [Notable Design Decisions](#11-notable-design-decisions)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)
    - [Open Follow-ups from the #47 Sweep](#open-follow-ups-from-the-47-sweep)
13. [Pitfalls and Gotchas](#13-pitfalls-and-gotchas)

---

## 1. Purpose and Big Picture

The repo implements an end-to-end pipeline that takes a photograph of a medieval chant manuscript folio and produces syllable-level bounding boxes annotated with Cantus Database text. The primary output is a JSON file consumable by the Pipeline Inspector GUI (and separately by the MEI encoding workflow). The end goal is aligning the neume notation in manuscript images with the syllabic text of the chant, enabling musicological research and digital edition work.

An older sub-project (YOLO/RTMDet segmentation model comparison) now lives in `experiments/` and is no longer part of the main pipeline.

**Artifact map:**

| Artifact | Entry point | Output |
|---|---|---|
| End-to-end pipeline (single folio) | `run_pipeline.py` | MEI JSON + optional Pipeline Inspector JSON |
| Multi-folio chaining wrapper | `run_chain.py` | Same outputs per folio, FolioState passed automatically |
| Pipeline Inspector GUI | `gui/` (React/Vite/OpenSeadragon) | Browser app, deployed to GitHub Pages |
| PAGE XML Viewer | `page_viewer.py` | Desktop Tkinter app |
| Utility scripts | `scripts/` | Various conversions and diagnostics |
| User documentation | `docs/` | `user_guide.md`, `user_decision_tree.md` |

---

## 2. Pipeline Stages

`run_pipeline.py::run()` executes six conceptual stages in sequence. Stage 0 (text-region masking) is handled at the top of `run()` itself, so it is available to all callers — CLI, `run_chain.py`, and the landing page backend.

### Stage 0 — Mothra Text-Region Masking (inside `run()`)

**Input:** raw folio image + optional mothra annotation JSON path  
**Output:** masked image written to a temp PNG; `run()` uses it for Stage 1, then deletes it

When `mothra_json_path` is passed to `run()`, masking is applied at the very start of `run()` using `MothraImageMask` (from `steps/mothra_mask.py`):

1. Reads the mothra annotation JSON and extracts all classId-1 (text) bboxes.
2. Builds a black canvas the same size as the folio image.
3. Paints a white rectangle around each bbox, expanded by `padding` pixels (default 15) on all sides.
4. Composites the original image through the mask — only text regions remain visible; everything else is black.
5. Saves the result to a temp PNG and passes it as `image_path` to `KrakenSegmentation`. A `try/finally` block ensures the temp file is deleted even if the pipeline raises an exception.

**Purpose:** BLLA over-segments on chant manuscripts partly because music notation (neumes, staves) creates spurious baselines. Blacking out non-text regions before segmentation suppresses the majority of false detections.

Masking lives inside `run()` so every caller — `main()`, `run_chain.py`, and the landing page — gets masking by simply passing `mothra_json_path`. When `mothra_json_path=None` (the default), the pipeline runs on the raw image unchanged. `--skip-masking` in the CLI prevents the resolved path from being passed to `run()`.

In the CLI (`main()`), `--mothra-json PATH` resolves and passes the path; `--skip-masking` suppresses it. In `run_chain.py`, `--mothra-jsons-dir DIR` looks for `{image_stem}.json` per folio and passes the path when found. Library callers (e.g. the landing page backend) pass it directly:

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

### Stage 1 — Kraken BLLA Line Segmentation (`KrakenSegmentation`)

**Input:** raw or masked folio image (JPEG/PNG/TIFF)  
**Output:** HTRflow `Collection` with line-level `SegmentNode` children

`KrakenSegmentation` wraps Kraken's BLLA (Baseline Line Detection Algorithm) as an HTRflow `PipelineStep`. For each page in the collection:

1. The HTRflow BGR numpy array is converted to PIL RGB (Kraken expects PIL).
2. `blla.segment()` runs inference — it detects baselines and builds polygon boundaries around each text line.
3. Lines with `boundary=None` are silently skipped (they cannot produce a valid `SegmentNode`).
4. A `Result.segmentation_result()` is produced with the polygon list.

**Custom model:** if `--segmentation-model` is passed, the model is loaded at construction time. `.safetensors` uses `load_safetensors`; `.mlmodel` (CoreML) uses `vgsl.TorchVGSLModel.load_model`. When `None`, Kraken's built-in BLLA model is used.

**Known issue with BLLA on chant manuscripts:** BLLA was trained on text manuscripts and tends to over-segment — drawing multiple bounding boxes around what is physically one text line because neume notation interrupts the text baseline. Stage 0 masking reduces this; Stage 2 fusion corrects remaining fragments.

### Stage 2 — Column Clustering and Co-linear Fusion (`cluster_columns` + `fuse_colinear_segments`)

**Input:** line nodes from Stage 1  
**Output:** reading-order label list, column count, split_x, and `FusedLine` objects

This stage has two sub-steps.

#### 2a. Column detection (`cluster_columns`)

Builds a 1D **coverage profile** array: `coverage[x]` = number of line bboxes that include pixel column `x`. The profile is smoothed with a 5-pixel window (to fill small polygon gaps). The search is scoped to the inner 20–80% of the text region (excluding blank margins).

**Bimodal test for two columns:**
- Find the valley (minimum) within the search band
- Find the peak on each side of the valley
- If `valley < bimodal_threshold × min(left_peak, right_peak)` AND gutter width ≥ 2% of page width AND each side has ≥15% of all lines → declare two columns at `split_x = gutter center`

With `forced_column_count=1`, sorting by ymin is returned immediately. With `forced_column_count=2`, if auto-detection fails, `split_x` falls back to the valley position (or page midpoint).

#### 2b. Spanning-bbox split

For two-column pages, any line bbox where `xmin < split_x < xmax` (with ≥10% overhang on each side) is split into two `_HtrflowSplitNode` objects — synthetic HTRflow leaf nodes that each wrap one half-crop. This corrects BLLA artefacts where a single bbox spans both columns.

#### 2c. Co-linear fusion (`fuse_colinear_segments`)

Groups BLLA sub-segments belonging to the same physical text line. Two segments are co-linear if their y-extents overlap by ≥50% of the shorter segment's height. Greedy grouping within each column produces `FusedLine` objects:

```
FusedLine(
    label="fused_0",                 # synthetic label
    constituent_labels=["region0_line2", "region0_line5"],  # original BLLA node labels
    constituent_widths=[312, 180],   # pixel widths
    xmin, xmax, ymin, ymax,          # bounding box of fused group
    column=1,                        # 1=left/only, 2=right
)
```

Fused lines are returned in reading order (left column top-to-bottom, then right column top-to-bottom). The fused labels (`fused_0`, `fused_1`, …) are used for NW alignment; `_defuse_manifest` in `run_pipeline.py` later redistributes words back to constituent labels proportionally by pixel width.

### Stage 3 — Kraken HTR Text Recognition (`KrakenRecognition`)

**Input:** line nodes  
**Output:** `node.text` populated for each line

For each active leaf node:
1. The HTRflow BGR crop is converted to PIL grayscale (Kraken HTR expects single-channel input).
2. A synthetic `Segmentation` with a single `BBoxLine` covering the full crop is constructed — we do NOT re-run BLLA; the crop is already the line region.
3. `rpred.rpred()` runs recognition and returns a `PredictionRecord`.
4. The prediction string and mean confidence are extracted and written back via `Result.text_recognition_result()`.

**Stub mode:** when `allow_stub=True` and `model=None`, all lines receive empty text and a WARNING is logged. Stub mode must be explicitly requested via `--stub-mode`; if no model is available and `--stub-mode` was not given, the CLI exits with an error rather than silently producing empty output.

**Tridis model:** the default recognition model is `Tridis_Medieval_EarlyModern.mlmodel`, a medieval/early-modern HTR model distributed via htrmopo (DOI `10.5281/zenodo.10788591`). At startup, `_find_tridis_model()` uses `platformdirs.user_data_dir("htrmopo")` to locate the model cache and globs for the filename across the UUID-named subdirectory. This is portable across machines.

**Known issue:** Tridis was trained on baseline segmentation crops. The mothra-text pipeline feeds it bbox crops (Kraken BLLA produces polygons but the node image is a bbox-extracted region). This type mismatch causes a "severely degraded performance" warning from Kraken. A chant-specific HTR model trained on bbox crops would resolve this.

### Stage 4 — NW Chant Allocator (`allocate_lines`) or OCR-Only Mode

**OCR-only mode** is triggered automatically when neither `--source-id` nor `--csv` is given. In this mode Stage 4 is skipped entirely: `manifest` is set to `{}` and all lines fall through to OCR-based word segmentation in Stage 5. The pipeline JSON is written with `"mode": "ocr_only"` instead of `"mode": "cantus_aligned"`. `--prev-folio-state` and `--folio-state-out` are ignored and a warning is logged.

In **Cantus-aligned mode**, this is the most complex stage. See Section 5 for a deep dive. At a high level:

1. `build_flat_text_and_anchors()` flattens all chant CSV rows for the folio into a single ordered word list, extracting volpiano break positions as `Anchor` objects.
2. `fuse_colinear_segments()` produces fused OCR texts (concatenated constituent texts).
3. **Pre-NW music-region filter** (optional, library callers only — not exposed as a CLI flag): when `music_boxes` (YOLO-detected stave/neume bounding boxes, absolute pixel `[x0, y0, x1, y1]` coordinates) is passed to `run()`, any fused line whose bbox overlaps a music region by more than `music_overlap_threshold` (default 0.30, strict `>`) of the line's own area is dropped from `sorted_labels` before NW alignment runs. This prevents spurious BLLA baselines near music staves from consuming ground-truth word slots and shifting every subsequent line off by one. Dropped nodes are recorded on `collection._music_filter_dropped`. Only `text-service/main.py` (the mothra API layer) passes `music_boxes`; CLI runs and `run_chain.py` pass `music_boxes=None` and are unaffected.
4. `allocate_lines()` maps each fused line label to a word span using Needleman-Wunsch alignment. Fused lines far too small to hold the text they are offered, whose OCR read nothing, are flagged `misdetected_line_skipped` and assigned no text so they cannot consume a line's words (see 5b). Unlike the music-region filter in step 3 this needs no external annotation — it works from the page's own line geometry — so it also protects CLI and `run_chain.py` runs.
5. `_defuse_manifest()` distributes words back to constituent node labels proportionally by pixel width. This is also why the veto in step 4 operates at the fused level: a tiny segment that y-overlaps a real line is *fused into* it and receives ~0 words from the proportional split, so only a tiny box that forms its own fused line can do damage.

**Output:** `manifest: dict[str, str]` — original node label → Cantus word fragment.

### Stage 5 — GT Word Segmentation (`GroundTruthWordSegmentation`)

**Input:** line nodes (with `node.text` from Stage 3); `gt_lookup` callable  
**Output:** word-level child nodes under each line node

For each active leaf (line node):
1. `gt_lookup(node)` returns the Cantus text fragment for that label from the manifest.
2. If GT text is available: `_bbox_word_segmentation()` divides the line's pixel width proportionally by character count (`pixels_per_char = width // len(text)`), creating one child node per word.
3. If GT text is empty/None: `_fallback_word_segmentation()` uses `node.text` (OCR output) instead. The word box's `source` field is set to `"fallback"` in `_build_pipeline_payload`.

The geometry is purely horizontal — word bboxes have the full line height and are stacked left-to-right without overlap.

### Stage 6 — Syllable Segmentation (`SyllableSegmentation`)

**Input:** word nodes  
**Output:** syllable-level child nodes under each word node

For each active leaf (word node):
1. `normalize_word_text()` strips non-ASCII, decomposes accents (NFKD), and lowercases.
2. `syllabify_word()` from `volpiano-display-utilities` determines syllable boundaries.
3. `_syllable_segmentation()` divides the word bbox character-proportionally across syllables.

Non-final syllables carry a trailing hyphen in the text field (e.g. `["do-", "mi-", "nus"]`). Single-syllable words and `LatinError` exceptions both produce a single child covering the full word.

---

## 3. File Structure

```
line-seg-eval/
├── run_pipeline.py             # main pipeline entry point (single folio)
├── run_chain.py                # multi-folio chaining wrapper
├── run_kraken.py               # standalone BLLA runner + visualizer
├── page_viewer.py              # PAGE XML viewer (Tkinter desktop app)
│
├── steps/                      # pipeline step implementations
│   ├── column_clustering.py    # stages 2a-2c
│   ├── gt_manifest.py          # Cantus CSV fetch + split_by_volpiano
│   ├── nw_chant_allocator.py   # stage 4 (NW alignment)
│   ├── ground_truth_word_segmentation.py  # stage 5
│   ├── syllable_segmentation.py           # stage 6
│   ├── kraken_segmentation.py             # stage 1
│   ├── kraken_recognition.py              # stage 3
│   ├── mothra_mask.py                     # stage 0 (text-region masking)
│   └── README.md
│
├── docs/                       # user-facing documentation
│   ├── user_guide.md           # troubleshooting and option reference
│   └── user_decision_tree.md   # GUI flag mapping + Mermaid decision tree
│
├── gui/                        # Pipeline Inspector browser app
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageCanvas.tsx # OSD viewer + SVG overlay
│   │   │   └── TopBar.tsx      # layer toggles + legend
│   │   ├── types.ts            # PipelineData, LineEntry, WordEntry, SyllableEntry
│   │   └── App.tsx
│   └── README.md
│
├── scripts/                    # utility / conversion scripts
│   ├── run_mothra_inference.py # run YOLOv11 mothra models → annotation JSON
│   ├── mothra_to_page.py       # Mothra Annotator JSON → PAGE XML
│   ├── convert_to_mei_input.py # pipeline JSON → MEI Text Alignment JSON
│   ├── compare_runs.py         # compare pipeline JSONs across approaches
│   ├── visualize_mothra.py     # overlay mothra bboxes on folio image
│   ├── debug_column_detection.py
│   └── README.md
│
├── experiments/                # comparative research (not the main pipeline)
│   ├── README.md
│   ├── run_htrflow.py          # YOLO/RTMDet runner
│   ├── run_all.py              # runs all three models
│   └── pipelines/              # htrflow YAML configs
│
└── tests/                      # pytest suite (200+ tests)
```

**Key design principle:** each pipeline step is a self-contained module in `steps/` that can be unit-tested independently. The HTRflow `Collection` tree is the shared data bus.

---

## 4. Framework Connections

### HTRflow

HTRflow provides the data model and step protocol. The repo uses:

- **`Collection`** — the root container; wraps a list of page images. Iterating yields `ImageNode` (page-level) objects.
- **`SegmentNode`** / **`ImageNode`** — tree nodes at page, line, word, and syllable levels. Key attributes: `bbox`, `polygon`, `image` (BGR numpy crop), `text`, `label`, `children`, `parent`.
- **`Collection.active_leaves()`** — yields the current bottom-level nodes (whatever level was last populated by a step).
- **`Collection.update(results)`** — appends a list of `Result` objects to the tree, creating the next level of children.
- **`Result.segmentation_result(shape, metadata, polygons=...)`** — creates line-level children.
- **`Result.text_recognition_result(metadata, texts, confidences)`** — sets `node.text`.
- **`Result.word_segmentation_result(orig_shape, metadata, bboxes, words)`** — creates word-level children.
- **`PipelineStep`** — base class with a `run(collection) -> collection` interface. All custom step classes inherit from this (or from its stub when the import fails).

**Apple Silicon stub pattern:** `htrflow.pipeline.steps` imports mmcv (for RTMDet) at module load time. On Apple Silicon, mmcv has a C extension symbol incompatible with the installed PyTorch, causing an `ImportError`. Every step file wraps its base class import in a `try/except` and defines a minimal stub. This is a htrflow framework issue, not something the repo introduced — the stubs must stay.

### Kraken

Used for both segmentation and recognition:
- **`kraken.blla.segment(pil_img, model=..., device=...)`** — BLLA inference; returns a `Segmentation` with `.lines` (each has `.boundary` polygon and `.baseline`).
- **`kraken.rpred.rpred(nn, pil_img, bounds)`** — HTR inference; takes a `Segmentation` (one `BBoxLine` per line crop), returns `PredictionRecord` list.
- **`kraken.lib.models.load_any(path, device=...)`** — loads `.mlmodel` or `.safetensors`.
- **`kraken.lib.vgsl.TorchVGSLModel.load_model(path)`** — alternate loader used for custom segmentation models.
- **`kraken.containers.BBoxLine`, `Segmentation`** — data containers passed to `rpred`.

### Bio.Align (Biopython)

Used in `nw_chant_allocator.py` for Needleman-Wunsch alignment:
- **`Bio.Align.PairwiseAligner`** — configured with `mode="global"`, match=8, mismatch=-5, open=-7, extend=-3. These parameters are calibrated for medieval Latin chant: high match reward, heavy gap penalty, moderate extend penalty.
- **`aligner.score(seq1, seq2)`** — returns raw alignment score without building the full traceback (fast path used in the scoring loop).
- **`aligner.align(seq1, seq2)`** — full alignment with traceback (used only in `debug=True` mode).

The "sequences" here are plain text strings (OCR text vs. candidate word spans), not DNA/protein sequences. Biopython treats them character-by-character.

### volpiano-display-utilities

Used in two places:
- **`syllable_segmentation.py`** — `syllabify_word(text, return_string=False)` → syllable boundary indices; `split_word_by_syl_bounds(text, bounds)` → list of syllable strings with trailing hyphens.
- **`nw_chant_allocator.py`** — `_split_word_at_syl_boundary()` uses the same functions to split words at volpiano-derived mid-line break points.

`LatinError` is raised for words that cannot be syllabified (e.g. proper names, abbreviations). Both callers catch it and fall back to single-syllable treatment.

### platformdirs

Transitive dependency of htrmopo. Used in `_find_tridis_model()` to locate the htrmopo model cache in a platform-appropriate way (`~/Library/Application Support/htrmopo` on macOS, `~/.local/share/htrmopo` on Linux, `%APPDATA%\htrmopo` on Windows).

### htrmopo

Zenodo-backed model manager. Not imported directly by pipeline code — used only via CLI (`python -m htrmopo get <doi>`). The UUID subfolder name is `uuid5(NAMESPACE_DNS, doi)`, which is why `_find_tridis_model()` globs for the filename rather than constructing the path.

---

## 5. NW Chant Allocator — Deep Dive

This is the most architecturally significant component. It solves the problem: given a sequence of detected text lines and a flat list of Cantus words, assign each line the words it physically contains.

### 5a. `build_flat_text_and_anchors()`

**Purpose:** flatten all chant CSV rows for a folio into a single ordered word list.

The Cantus CSV has one row per chant; each row contains `fulltext_ms` (manuscript spelling) and `volpiano` (neume notation encoding). The volpiano uses structural break markers:

| Marker | Meaning |
|---|---|
| `7` | within-chant line break |
| `77` | page break (chant continues on next folio) |
| `777` | column break |
| `---` | word boundary within volpiano |
| `--` | syllable boundary within a word |

`_parse_row_words_and_anchors()` processes each row's text+volpiano:
- Splits volpiano on `7+` to get segments
- For each segment, counts word groups (stretches between `---` with at least one note letter)
- Detects mid-word line breaks (when a segment after `7` starts with `--`, its first group is a mid-word continuation, not a new word)
- Records `(word_index, anchor_type)` pairs for each break
- When a `77` is found, all words after it are `continuation_words` for the next folio

After processing all folio rows, `FlatTextData` contains:
- `words`: flat list of all words on this folio
- `anchors`: `Anchor(word_index, anchor_type)` list — positions of line breaks
- `chant_spans`: `ChantSpan(sequence, start_word, end_word)` list — which words belong to which chant
- `mid_word_breaks`: `MidWordBreak(anchor_word_index, syl_left, syl_right)` for breaks mid-word
- `continuation_words`: words physically on the next folio (no separate CSV row for them)
- `initial_pointer`: start position for NW alignment (0 unless continuation was prepended)

**Continuation handling:** when `prev_folio_state` is provided, its `remaining_words` are prepended to `flat_text.words`. When it's absent, the code scans preceding CSV rows for the last row with a `77` break and automatically infers the carry-over (`infer_continuation=True`). If no continuation is found and the first word of the folio is lowercase, a `continuation_missing` validation flag is emitted.

**Suffix probe words:** for the no-volpiano folio-start-location feature — words from the preceding folio's last chant stored for use in `pre_start_suffix_align`.

### 5b. `allocate_lines()` — the NW allocation loop

**Algorithm overview:**

For each fused line label in reading order:

1. **Candidate words:** `flat_text.words[text_pointer : text_pointer + search_window]`
2. **NW scoring:** for k = 1 … len(candidate_words), score `aligner.score(ocr, " ".join(words[:k]))` normalized by geometric mean of string lengths. Pick k with best normalized score. The normalization `raw / sqrt(len(ocr) * len(window))` corrects for the tendency to prefer longer candidates.
3. **Anchor snap (snap_window=2):** if the NW end `text_pointer + k` is within 2 words of the next `within_chant_7` or `page_break_77` anchor, snap to the anchor silently. Rationale: NW is almost right but anchors are ground truth.
4. **Force window (force_window=10):** if NW is wrong by more than snap_window but within 10 words, AND the anchor is `within_chant_7` (not page breaks), AND no new chant span starts in the gap → force the anchor anyway. Emits a `forced_mid_chant_snap` flag. Rationale: mid-chant anchors are structurally reliable; OCR errors in a bad stretch should not cascade.
5. **Disagreement flag:** if neither snap nor force applies, a `nw_volpiano_disagreement` (or `page_break_77_mismatch`) flag is emitted.

**snap_window vs. force_window — they are not redundant:**
- `snap_window`: NW was nearly right → fine-tune silently
- `force_window`: NW was wrong (bad OCR stretch) → override with structural knowledge, but only within-chant and only when no new chant starts in the gap (which would mean NW's evidence about which physical line a chant starts on could be valid)

**Stub mode (empty OCR):** each line advances to the next anchor of any type. When no anchors are available, remaining words are distributed uniformly across remaining lines.

6. **Misdetected-line veto (`skip_misdetected_lines`, default on):** evaluated on the final NW/stub demand, before the col-1 clamp and force-close. Stub mode above is designed for the whole-page case (`--stub-mode`); when a *single* line is empty on a page of good OCR it is the wrong behaviour, because that line is usually not a line at all — BLLA drew a box around a neume group, a clef sliver, or an initial. The veto fires only when all four of these agree: allocation wants ≥ `misdetect_min_words` words (default 1); OCR holds < `misdetect_min_ocr_chars` **alphanumeric** characters (default 2, so `'„'` counts as nothing); the box is narrower than `misdetect_width_ratio` of the page's typical line (default 0.35); and its character capacity is below `misdetect_capacity_ratio` of the text being assigned (default 0.4). The line is then assigned `""`, recorded in `skipped_labels`, flagged `misdetected_line_skipped`, and `continue`d — leaving `text_pointer` *and* `syllable_prefix` untouched so the words, and any carried mid-word fragment, go to the next line. Page statistics come from `_page_line_scale()`: the median width of lines whose OCR has ≥ 8 alphanumeric characters, and the *narrowest* average character width among them (the most generous capacity estimate). Fewer than 3 such lines → no page scale → feature inert, which also covers global stub mode. The veto runs in this main loop and in both pre-start sub-branches, where a vetoed box is additionally exempt from the force-snap.

Conditions 3 and 4 are both required because either alone misfires: width alone would veto a genuine short closing line whose OCR failed, and demand alone would veto a real short line that stub mode over-fed. Conversely `misdetect_min_words` is 1, not 2 — a word *count* cannot distinguish a 379px box that can legitimately hold `in` from a 69px box that cannot hold `iactatus`; only the capacity test can, so gating on word count would wrongly spare the latter.

**Two-column handling:** at `label_idx == left_column_count`, the text pointer is hard-reset to the `column_break_777` anchor word index. The last col-1 line is force-closed at that anchor too, so no words fall through the gap.

**Mid-word breaks:** when `text_pointer + consumed` lands on a `MidWordBreak`, the last word of the current line is split at the volpiano syllable boundary using `_split_word_at_syl_boundary()`. The right fragment is stored in `syllable_prefix` and prepended to the next line's word list.

**No-volpiano folio-start location (`locate_folio_start`):** when `flat_text.anchors` is empty (all chants lack volpiano), the code searches for which line best matches the first folio chant via NW, declares that as line index L*, and treats lines 0…L*-1 as pre-start (previous folio's bleeding continuation). Pre-start lines receive either continuation words (via NW), suffix alignment words (preceding folio's last chant tail), or empty strings.

**Mixed-line detection:** for the last pre-start line (index L*-1), the code checks whether the rightmost constituent(s) of that fused line contain any opening words of this folio's first chant. If so, it constructs per-constituent `constituent_overrides` that correctly split the line between "previous folio continuation" and "this folio start".

**Output:** `AllocationResult(manifest, flags, text_pointer_end, debug_lines, folio_start_line, constituent_overrides, skipped_labels)`

### 5c. `_defuse_manifest()`

After NW allocation, the manifest has fused labels (`fused_0`, `fused_1`, …). `_defuse_manifest` distributes each fused line's word list back to its constituent node labels proportionally by pixel width. The last constituent receives any remainder words to avoid rounding loss.

### 5d. Folio state persistence

`FolioState` carries `remaining_words` (post-77 continuation) and `last_chant_sequence` across folio runs. Written to a JSON sidecar via `write_folio_state()`; read back via `read_folio_state()`. Used to chain pipeline runs across consecutive folios of the same manuscript.

For manual runs, `--folio-state-out` / `--prev-folio-state` manage this by hand. For automated batch processing, `run_chain.py` handles it transparently (see Section 8).

---

## 6. Key Data Structures

### HTRflow tree hierarchy

```
Collection
└── ImageNode (page)                       label="image_0"
    └── SegmentNode (line)                 label="region0_line0"  ← after Stage 1
        └── SegmentNode (word)             label="region0_line0_word0"  ← after Stage 5
            └── SegmentNode (syllable)     label="region0_line0_word0_syl0"  ← after Stage 6
```

Each `SegmentNode` has:
- `bbox`: `Bbox(xmin, ymin, xmax, ymax)` — absolute coordinates in page image space
- `polygon`: polygon in page image space (only for line nodes from BLLA)
- `image`: BGR numpy array crop (lazy-loaded from parent)
- `text`: string (set after Stage 3 for lines; word/syllable text set by segmentation steps)
- `label`: string (dot-separated path, e.g. `region0_line0`)
- `width`, `height`: derived from bbox

### FlatTextData

Produced by `build_flat_text_and_anchors()`:
```python
@dataclass
class FlatTextData:
    words: list[str]                    # all words on the folio, in order
    anchors: list[Anchor]               # volpiano break positions
    chant_spans: list[ChantSpan]        # which words belong to which chant
    mid_word_breaks: list[MidWordBreak] # breaks that fall mid-word
    initial_pointer: int                # start index for NW alignment
    continuation_words: list[str]       # post-77 words for next folio
    has_continuation: bool              # True when prev-folio words were prepended
    suffix_probe_words: list[str]       # preceding folio's last chant words
```

### Pipeline JSON output schema

```json
{
  "folio": "CH-E_611_001r",
  "mode": "cantus_aligned",
  "image_width": 2480,
  "image_height": 3508,
  "lines": [
    {
      "label": "region0_line0",
      "bbox": [x0, y0, x1, y1],
      "polygon": [[x, y], ...],
      "text": "",
      "words": [
        {
          "label": "region0_line0_word0",
          "text": "Alleluia",
          "bbox": [x0, y0, x1, y1],
          "source": "gt",
          "syllables": [
            {
              "label": "region0_line0_word0_syl0",
              "text": "Al-",
              "bbox": [x0, y0, x1, y1]
            }
          ]
        }
      ]
    }
  ]
}
```

**`mode`** is `"cantus_aligned"` when a Cantus source was supplied or `"ocr_only"` when neither `--source-id` nor `--csv` was given. The GUI and downstream tools use this field to adjust rendering and processing behaviour.

**`folio`** is the regularized identifier when `--output-dir` is used (e.g. `CH-E_611_001r`), derived from the Cantus CSV's RISM code and shelfmark. When `--export-json` is used without `--output-dir`, it defaults to the image filename stem.

The `source` field is `"gt"` when the manifest has a non-empty truthy value for that line's label, and `"fallback"` when the line fell through to OCR-based word segmentation.

### Mothra annotation JSON schema

Produced by `scripts/run_mothra_inference.py` and consumed by `MothraImageMask` and the upstream masking step:

```json
{
  "imageName": "001r.jpg",
  "imageWidth": 2480,
  "imageHeight": 3508,
  "annotations": [
    {
      "id": 1,
      "classId": 1,
      "bbox": [x, y, width, height],
      "confidence": 0.87,
      "timestamp": "2026-07-06T..."
    }
  ]
}
```

`classId` values: `1` = text, `2` = music, `3` = staves. `MothraImageMask` uses only classId-1 bboxes. `bbox` is in `[x_topleft, y_topleft, width, height]` format (absolute pixels).

---

## 7. GUI Architecture

The Pipeline Inspector GUI (`gui/`) is a React/Vite/TypeScript browser app deployed to GitHub Pages. It uses **OpenSeadragon (OSD)** for deep-zoom tile rendering of folio images and an **SVG overlay** for annotation rendering.

**Key components:**

- **`App.tsx`** — manages state: `imageUrl`, `data` (parsed JSON), `layers` visibility toggles. Handles file opening via two `<input type="file">` elements.
- **`ImageCanvas.tsx`** — the main canvas. Mounts an OSD viewer on a `<div>`. A `<svg>` absolutely positioned on top renders all annotation layers. On every OSD `update-viewport` event, all annotation coordinates are recomputed from image-space to viewer-element-space via `viewer.viewport.imageToViewportCoordinates()` + `viewer.viewport.viewportToViewerElementCoordinates()`. Polygon `points` strings are recalculated on every pan/zoom event.
- **`TopBar.tsx`** — layer toggle buttons (Lines/Words/Syllables with counts) and color legend (GT teal / OCR fallback rose). Legend is only shown when the Words layer is active.

**Rendering layers (bottom-to-top in SVG):**
1. Line polygons — purple, clickable (click to see line text in a popup panel)
2. Syllable rects — amber/orange, with text labels when box is wide enough
3. Word rects — teal for GT, rose for fallback, with text labels when box is wide enough

**Word source coloring:**
- GT (`source="gt"`): `rgb(45,212,191)` teal fill + stroke
- Fallback (`source="fallback"`): `rgb(251,113,133)` rose fill + stroke

The SVG `pointer-events: none` allows OSD pan/zoom to work through the overlay. Line polygons have `pointer-events: all` specifically so they can be clicked for the line text popup.

---

## 8. Multi-Folio Chaining (`run_chain.py`)

`run_chain.py` is a wrapper around `run_pipeline.run()` that processes a sequence of consecutive folios in order, passing `FolioState` between runs automatically so the user does not need to manage `--folio-state-out` / `--prev-folio-state` sidecar files by hand.

**Key behaviours:**

- Requires `--images` and `--folios` lists of equal length (≥2 folios required; use `run_pipeline.py` directly for a single folio).
- `--source-id` or `--csv` is required (mutually exclusive); OCR-only mode is not available in `run_chain.py`.
- `--export-json` accepts one output path per folio (same order as `--images`).
- `--output-dir PATH` auto-names MEI JSON outputs as `{RISM-code}_{shelfmark}_{folio}.json` using `make_output_stem()`. Explicit per-folio paths via `--mei-json` take precedence over `--output-dir`.
- `--folio-states-dir PATH` saves intermediate `FolioState` JSON files as `state_{folio}.json` for post-run inspection. Omit to discard intermediate states.
- `--mothra-jsons-dir PATH` enables text-region masking for the chain. For each folio, `run_chain.py` looks for `{image_stem}.json` in the directory and passes the path to `run()` when found. A missing JSON for any folio produces a warning and that folio runs unmasked. The directory is typically the `--out-dir` of a prior `scripts/run_mothra_inference.py` run.
- `--padding PX` (default 15) controls bbox expansion during masking, passed through to `run()`.
- `--skip-masking` suppresses masking even when `--mothra-jsons-dir` is provided.
- If any folio fails, the chain aborts and reports how many folios completed before the failure.

**Internal flow** per folio:
1. Resolve the mothra JSON path for this folio from `--mothra-jsons-dir` (or `None` if absent or missing).
2. Call `run_pipeline.run()` with `prev_state` and `mothra_json_path` — masking is applied inside `run()`.
3. Write optional Pipeline Inspector JSON and MEI JSON.
4. Read the `FolioState` from a temp file (written by `run()`).
5. Log continuation word count and `fully_consumed` status.
6. Optionally copy temp state to `folio_states_dir`.
7. Pass `next_state` as `prev_state` for the next iteration.

---

## 9. Mothra Text-Detection Integration

### `MothraImageMask` (`steps/mothra_mask.py`)

The masking approach pre-processes the folio image so Kraken BLLA only sees text regions. It does not modify the `Collection` tree or inject nodes — it produces a masked image that is passed to `KrakenSegmentation` in place of the original.

```python
masker = MothraImageMask(mothra_json_path, padding_px=15)
masked_pil = masker.apply(original_pil_image)
```

The `padding_px` parameter (CLI default: 15, class default: 25) controls how much each text bbox is expanded before masking. Larger values merge nearby word-level detections into line-width strips that Kraken can detect as full lines. Smaller values reduce bleed into adjacent music rows but may leave gaps within long text lines. The optimal value depends on manuscript layout — tightly packed manuscripts may need `--padding 10`; manuscripts with wider word spacing may need `--padding 20`.

### `scripts/run_mothra_inference.py`

Produces mothra annotation JSONs from raw folio images using YOLOv11 models hosted on HuggingFace Hub. This is the upstream step that generates the JSON consumed by `MothraImageMask`.

**Usage:**
```
python scripts/run_mothra_inference.py \
    --images 001r.jpg 001v.jpg 002r.jpg \
    --out-dir ~/Downloads/mothra_jsons/ \
    --conf 0.25
```

**What it runs:**
- `text_music_detector` model → classId 1 (text regions), classId 2 (music regions)
- `stave_detector` model → classId 3 (stave regions)

Models are downloaded from HuggingFace Hub on first run (requires a HF token at `~/.cache/huggingface/token`). Images where an output JSON already exists are skipped.

### `scripts/visualize_mothra.py`

Overlays mothra detection bboxes on a folio image for visual inspection. Useful for checking mothra JSON quality before running the full pipeline.

```
python scripts/visualize_mothra.py mothra.json folio.jpg output.jpg
```

Color coding: text (classId 1) = green, music (classId 2) = blue, staves (classId 3) = red.

### `scripts/compare_runs.py`

Compares pipeline export JSONs across different runs (e.g. baseline vs. masked) and prints a table of line counts and word-source statistics (GT vs. fallback words).

```
python scripts/compare_runs.py \
    --label baseline ~/Downloads/DDMAL/baseline_12v.json \
    --label masked   ~/Downloads/DDMAL/mothra_masked_12v.json \
    --output ~/Downloads/DDMAL/mothra_comparison_report_2026-07-13.txt
```

---

## 9a. Relationship to the `mothra` Repo (Landing Page / text-service)

Section 9 above covers a *different* axis of "mothra integration" — this repo *consuming* mothra's
YOLO annotation-JSON format via `MothraImageMask`/`scripts/run_mothra_inference.py`. This section
covers the *other* direction: how the separate [`mothra`](https://github.com/DDMAL/mothra) repo
(the landing-page application) invokes `mothra-text` in production.

`mothra` includes `mothra-text` as a git submodule pinned at a specific commit. The landing
page's PostgreSQL `annotations` table (populated by its own YOLO inference step) feeds
`landing-page/scripts/text_api.py`, which builds a mask JSON (equivalent to this repo's
`--mothra-json`) and a music-box list from the stored detections. Those cross an HTTP boundary —
a POST to `text-service` on `:8002/run` — where `text-service/main.py` writes the mask JSON to a
temp file and calls this repo's `run()` directly. From that point on, the integrated path and the
CLI path are running identical code on identical inputs. Results are written to the landing
page's `text_alignments` table.

One asymmetry worth knowing: `text-service` performs its own post-`run()` music-box filter step
(`filter_lines_over_music()`) that does **not** exist anywhere in `mothra-text`'s own CLI or
library surface — don't assume the CLI fully represents production behavior when debugging a
discrepancy between a local run and the live landing page.

See the confidence-threshold coordination note in [§13 Pitfalls and Gotchas](#13-pitfalls-and-gotchas)
for a concrete example of how these two paths can silently diverge.

**This section is intentionally a short summary, not the full picture** — the authoritative
architecture diagram and write-up (PostgreSQL schema, the exact `text_api.py` mask/music-box
construction, coordinate translation between YOLO's normalized floats and `MothraImageMask`'s
pixel integers, and the full behavior-comparison table) lives at
[DDMAL/mothra#151](https://github.com/DDMAL/mothra/issues/151). Keep it that way — duplicating
the full write-up here would create exactly the kind of doc/code drift this sweep (mothra-text#47)
was written to fix.

---

## 10. Tools and External Dependencies

| Dependency | What it does | Where used |
|---|---|---|
| `kraken` | BLLA line segmentation; Kraken HTR recognition | `KrakenSegmentation`, `KrakenRecognition` |
| `htrflow` | Collection/SegmentNode tree; Result objects; PipelineStep base | all steps |
| `biopython` | Needleman-Wunsch alignment via `Bio.Align.PairwiseAligner` | `nw_chant_allocator.py` |
| `volpiano-display-utilities` | Latin syllabification | `syllable_segmentation.py`, `nw_chant_allocator.py` |
| `platformdirs` | Cross-platform user data directory (transitive dep of htrmopo) | `run_pipeline.py`, `run_chain.py` |
| `numpy` | 1D coverage profile array; convolution smoothing | `column_clustering.py` |
| `opencv-python (cv2)` | BGR↔RGB conversion for PIL/Kraken | `KrakenSegmentation`, `KrakenRecognition` |
| `Pillow (PIL)` | Image masking (`MothraImageMask`); Kraken input; `page_viewer.py` | `mothra_mask.py`, multiple |
| `ultralytics` | YOLOv11 inference for mothra text-region detection | `scripts/run_mothra_inference.py` only |
| `htrmopo` | Model download/caching (used via CLI only) | CLI workflow |

**Standard library:** `argparse`, `csv`, `io`, `json`, `logging`, `pathlib`, `re`, `shutil`, `statistics`, `tempfile`, `unicodedata`, `urllib.request`.

**GUI deps (npm):** React, TypeScript, Vite, Tailwind CSS, OpenSeadragon, `@types/openseadragon`.

---

## 11. Notable Design Decisions

### HTRflow as scaffolding, not as the core

HTRflow provides the `Collection` tree and step protocol but contributes zero domain logic. All of the chant-specific behavior — co-linear fusion, NW alignment, syllabification, GT word segmentation — is implemented in `steps/`. HTRflow could be replaced with a different orchestration layer without touching the domain logic.

### NW alignment over direct volpiano line-splitting

The earlier approach (`build_page_manifest()` in `gt_manifest.py`) split chant text directly by volpiano `7` markers and zipped the result with node labels. This fails when chants start mid-line (previous chant ends partway through a physical line) — the zip assumes a clean one-to-one correspondence. The NW allocator avoids this assumption entirely: it uses OCR text as evidence to determine where each line starts and ends in the flat word sequence.

### Masking before segmentation, not injection after

An earlier approach (`MothraUnionStep`, now removed) injected mothra-detected lines that Kraken missed into the HTRflow collection after segmentation. This was fragile: injected nodes had no polygon, required deduplication against Kraken nodes via IoU, and produced uneven line coverage that confused NW alignment. The current approach masks the image before Kraken runs. Kraken then detects only text regions and produces polygons for all of them — the tree is uniformly populated with real Kraken nodes.

### Masking inside `run()`, not in callers

`MothraImageMask` writes a temp PNG, substitutes it for `image_path` during Stage 1, then deletes it in a `try/finally` block. Because masking lives inside `run()`, every caller — `main()`, `run_chain.py`, and the landing page — gets masking for free by passing `mothra_json_path`. No caller needs to manage temp files or know about PIL.

### OCR-only mode as the default when no Cantus data is given

Rather than requiring the user to pass an explicit `--ocr-only` flag, the mode is inferred: if neither `--source-id` nor `--csv` is given, Cantus alignment is skipped automatically. This makes the CLI easier to use for quick inspection runs while keeping the full Cantus-aligned mode as the documented primary workflow.

### Fused-line NW, constituent-label manifest

NW alignment runs on fused lines (one logical text line = potentially multiple BLLA segments). The fused-line OCR text is the concatenation of constituent OCR texts. After allocation, `_defuse_manifest` distributes words back to constituent labels proportionally by pixel width. This means the fused grouping is transparent to all downstream steps (Stages 5 and 6 operate on original node labels).

### Source field truthiness, not key presence

`_build_pipeline_payload` marks a word as `source="gt"` using `manifest.get(line_node.label)` (falsy check), not `line_node.label in manifest` (key existence). These diverge for lines that received an empty-string GT assignment (e.g. a line with no text in the Cantus CSV). `GroundTruthWordSegmentation` also uses `if not gt_text:` to trigger fallback. The two checks must agree — an empty-string assignment means no GT text was available, so the word should be `"fallback"`.

### Regularized output naming via `make_output_stem()`

`--output-dir` triggers `make_output_stem(csv_rows, folio)` which extracts the RISM code and shelfmark from the Cantus CSV (fields already downloaded for alignment) and constructs a standardized filename like `CH-E_611_001r.json`. This avoids ad-hoc naming and keeps MEI JSON outputs directly consumable by downstream tools without renaming.

### Apple Silicon stub pattern

Every step that inherits from `htrflow.pipeline.steps` wraps the import in `try/except ImportError` and defines a local stub. This is not defensive programming for general failure — it is specifically for the Apple Silicon mmcv crash. The stubs are load-time shims, not runtime fallbacks. They should never be removed until HTRflow stops importing mmcv at module level.

---

## 12. Known Limitations and Future Work

| Area | Issue | Path forward |
|---|---|---|
| HTR model | Tridis was trained on baseline crops; we feed bbox crops → "severely degraded performance" warning | Train a chant-specific HTR model on bbox crops |
| Segmentation | BLLA over-segments on neumes; co-linear fusion is a post-hoc fix; masking helps but doesn't eliminate the problem | Fine-tune BLLA on chant manuscripts (Kraken BLLA fine-tuning setup exists in the repo) |
| NW with no volpiano | Folio-start location is approximate; works at line granularity | No clear path; volpiano is the ground-truth structural signal |
| Word geometry | Character-proportional bbox distribution ignores actual character widths | Would require OCR character-level bounding boxes from Kraken (possible but not currently extracted) |
| GUI | Deployed build is static; updating requires `npm run build` + push | Already automated via GitHub Actions for `gui/` changes |

### Open Follow-ups from the #47 Sweep

[mothra-text#47](https://github.com/DDMAL/mothra-text/issues/47) was a full-codebase
production-quality sweep (docs, tests, tidiness, CI). Everything below was deliberately
**not** done as part of it — noted here so it isn't mistaken for an oversight:

- **argparse boilerplate duplication** between `run_pipeline.py`'s and `run_chain.py`'s
  `main()` (~10 near-identical flags: `--segmentation-model`, `--recognition-model`,
  `--stub-mode`, `--device`, `--column-bimodal-threshold`, `--column-count`, `--debug-ocr`,
  `--padding`, `--skip-masking`, `--output-dir`). Deferred because deduplicating touches both
  CLI entry points' help text and defaults simultaneously; a follow-up should add
  argparse-output snapshot tests before refactoring, not after.
- **`_find_tridis_model()` duplication was investigated, not fixed.** Importing it from
  `run_pipeline.py` into `run_chain.py` was attempted, then reverted: `run_pipeline.py`
  unconditionally imports htrflow/kraken/torch at module level, so the import alone added
  ~5 seconds to every `run_chain.py` invocation, including `--help`. The two copies remain
  deliberately duplicated (see both functions' docstrings); a real fix would need extracting
  the lookup into a new dependency-free module, which felt like more structural change than
  this sweep should make unilaterally.
- **`run_kraken.py` relocation to `experiments/`** — it's a segmentation-comparison baseline,
  organizationally misplaced at repo root next to `run_pipeline.py`/`run_chain.py`, but moving
  it has import-path implications for `experiments/run_all.py` and wasn't treated as
  mechanical enough to do without a deliberate decision.
- **GUI test framework bootstrap and all GUI test coverage.** `gui/` has zero test
  infrastructure (no framework installed, no `test` script in `package.json`) — a capability
  gap, not a coverage gap. Choosing Vitest vs. Jest, a DOM shim, React Testing Library
  conventions, and CI wiring deserves its own scoped issue and review rather than being
  bundled into an already Python-heavy sweep. **A follow-up issue should be filed for this**
  (not yet done as of this sweep).
- **`run_kraken.py` and `scripts/*.py` unit test coverage** — deferred as lower risk (neither
  is in the request-serving path).
- **`run_pipeline.py`'s full `run()` orchestration** (`export_json`, `_defuse_manifest`,
  `_split_spanning_nodes_in_tree`, and `run()` end-to-end) remains untested beyond the pure
  `_music_overlap_ratio` helper and the music-filter block (Tier 1 of the #47 test sweep,
  `tests/test_run_pipeline.py`) — full orchestration coverage needs either heavy
  Kraken/HTRflow/Tridis mocking or a lightweight extraction of testable sub-functions,
  neither attempted here.
- **The design gap behind the mothra-text/mothra confidence-threshold coordination pitfall**
  (see [§13](#13-pitfalls-and-gotchas)) is documented and given a regression-guarding test
  (`tests/test_mothra_mask.py`), but the underlying fix — deciding whether `MothraImageMask`
  should start reading/filtering on confidence itself, rather than trusting the upstream
  JSON unconditionally — was not made. That's a behavior change requiring cross-repo
  coordination with `mothra`'s maintainers, out of scope for a docs/tests/tidy issue.

---

## 13. Pitfalls and Gotchas

Non-obvious behavior and easy-to-conflate concepts that have caused (or could cause) real
confusion for a future developer or an LLM working in this codebase.

### `snap_window` / `force_window` now share one default (2/10)

`allocate_lines()`'s `snap_window`/`force_window` parameters (see [§5b](#5b-allocate_lines--the-nw-allocation-loop))
default to `2`/`10` — the same tolerance values the one production call site
(`run_pipeline.py`) actually runs with. This matters because `allocate_lines()` is a public
function other callers (tests, a future integration) might invoke directly: before this was
unified, a direct caller silently got weaker snapping (`force_window=0`, i.e. mid-chant
force-snap disabled) than the CLI, with nothing in the code or docs flagging the difference.

### Confidence-threshold coordination between mothra-text and `mothra`

`MothraImageMask.apply()` (`steps/mothra_mask.py`) never reads or filters on the mothra JSON's
confidence field — it trusts whatever detections are in the JSON, whole-cloth, regardless of
score. Until 2026-08-18, this was a *live, confirmed* discrepancy: `scripts/run_mothra_inference.py`
defaults `--conf` to `0.25`, while the `mothra` (landing-page) repo's production inference
defaulted to `0.5` — meaning detections scoring 0.25–0.49 were silently present in CLI-generated
masks but absent from production ones (root-caused in
[mothra#151](https://github.com/DDMAL/mothra/issues/151)). It was fixed in
[mothra#244](https://github.com/DDMAL/mothra/pull/244), which changed the production default to
match mothra-text's `0.25`. **The underlying design gap remains**: `mothra-text` has no way to
detect a threshold mismatch on its own side. If the two repos' defaults ever drift apart again,
nothing here will notice — this is a coordination point to keep in mind, not a solved problem.

### `--column-bimodal-threshold` vs. YOLO `--conf` — different systems, similar names

`--column-bimodal-threshold` (default `0.5`, `column_clustering.py`) controls the two-column
detection valley/peak ratio. YOLO `--conf` (default `0.25`, `scripts/run_mothra_inference.py`)
controls detection confidence filtering. They are unrelated systems that both get called
"threshold"/"conf" in nearby contexts (masking and layout discussions) — easy to conflate when
skimming code or docs; check which module a bare "threshold" mention actually refers to.

### Two incompatible bbox coordinate formats

Mothra annotation JSON bboxes are `[x, y, width, height]` (see the
[Mothra annotation JSON schema](#mothra-annotation-json-schema)), but this pipeline's own JSON
output and the `music_boxes` parameter both use `[x0, y0, x1, y1]` absolute corners (see
[§2 Stage 4](#stage-4--nw-chant-allocator-allocate_lines-or-ocr-only-mode) and the
[Pipeline JSON output schema](#pipeline-json-output-schema)). Each format is documented correctly
in its own section, but nothing previously flagged them together — don't assume a `bbox`/`box`
value uses one format just because a nearby one does; check which schema you're actually in.
