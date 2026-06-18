# mothra-text Deep Dive Report

**Repo:** `/Users/cassiebastress/Documents/DDMAL/line-seg-eval`
**Date written:** 2026-06-18

---

## 1. Purpose and Big Picture

The repo implements an end-to-end pipeline that takes a photograph of a medieval chant manuscript folio and produces syllable-level bounding boxes annotated with Cantus Database text. The primary output is a JSON file consumable by the Pipeline Inspector GUI (and separately by the MEI encoding workflow). The end goal is aligning the neume notation in manuscript images with the syllabic text of the chant, enabling musicological research and digital edition work.

Two older sub-projects (YOLO/RTMDet segmentation model comparison and PyLaia HTR baselines) now live in `experiments/` and are no longer part of the main pipeline.

**Artifact map:**

| Artifact | Entry point | Output |
|---|---|---|
| End-to-end pipeline | `run_pipeline.py` | JSON + optional MEI JSON |
| Pipeline Inspector GUI | `gui/` (React/Vite/OpenSeadragon) | Browser app, deployed to GitHub Pages |
| PAGE XML Viewer | `page_viewer.py` | Desktop Tkinter app |
| Utility scripts | `scripts/` | Various conversions |

---

## 2. Pipeline Stages

`run_pipeline.py::run()` executes six conceptual stages in sequence. Each stage operates on an HTRflow `Collection` object (a tree of image nodes) or produces a data structure consumed by the next stage.

### Stage 1 — Kraken BLLA Line Segmentation (`KrakenSegmentation`)

**Input:** raw folio image (JPEG/PNG/TIFF)  
**Output:** HTRflow `Collection` with line-level `SegmentNode` children

`KrakenSegmentation` wraps Kraken's BLLA (Baseline Line Detection Algorithm) as an HTRflow `PipelineStep`. For each page in the collection:

1. The HTRflow BGR numpy array is converted to PIL RGB (Kraken expects PIL).
2. `blla.segment()` runs inference — it detects baselines and builds polygon boundaries around each text line.
3. Lines with `boundary=None` are silently skipped (they cannot produce a valid `SegmentNode`).
4. A `Result.segmentation_result()` is produced with the polygon list.

**Custom model:** if `--segmentation-model` is passed, the model is loaded at construction time. `.safetensors` uses `load_safetensors`; `.mlmodel` (CoreML) uses `vgsl.TorchVGSLModel.load_model`. When `None`, Kraken's built-in BLLA model is used.

**Known issue with BLLA on chant manuscripts:** BLLA was trained on text manuscripts and tends to over-segment — drawing multiple bounding boxes around what is physically one text line because neume notation interrupts the text baseline. Stage 2 corrects this.

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

**Stub mode:** when `model=None`, all lines receive empty text and a WARNING is logged. The pipeline continues normally — all downstream steps are designed to tolerate empty OCR text.

**Tridis model:** the default recognition model is `Tridis_Medieval_EarlyModern.mlmodel`, a medieval/early-modern HTR model distributed via htrmopo (DOI `10.5281/zenodo.7899855`). At startup, `_find_tridis_model()` uses `platformdirs.user_data_dir("htrmopo")` to locate the model cache and globs for the filename across the UUID-named subdirectory. This is portable across machines.

**Known issue:** Tridis was trained on baseline segmentation crops. The mothra-text pipeline feeds it bbox crops (Kraken BLLA produces polygons but the node image is a bbox-extracted region). This type mismatch causes a "severely degraded performance" warning from Kraken. A chant-specific HTR model trained on bbox crops would resolve this.

### Stage 4 — NW Chant Allocator (`allocate_lines`)

This is the most complex stage. See Section 5 for a deep dive. At a high level:

1. `build_flat_text_and_anchors()` flattens all chant CSV rows for the folio into a single ordered word list, extracting volpiano break positions as `Anchor` objects.
2. `fuse_colinear_segments()` produces fused OCR texts (concatenated constituent texts).
3. `allocate_lines()` maps each fused line label to a word span using Needleman-Wunsch alignment.
4. `_defuse_manifest()` distributes words back to constituent node labels proportionally by pixel width.

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
├── run_pipeline.py             # main pipeline entry point
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
│   └── README.md
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
│   ├── mothra_to_page.py       # Mothra Annotator JSON → PAGE XML
│   ├── convert_to_mei_input.py # pipeline JSON → MEI Text Alignment JSON
│   ├── debug_column_detection.py
│
├── experiments/                # comparative research (not the main pipeline)
│   ├── README.md
│   ├── run_htrflow.py          # YOLO/RTMDet runner
│   ├── run_all.py              # runs all three models
│   ├── pipelines/              # htrflow YAML configs
│   └── pylaia_baseline/        # PyLaia HTR experiments
│       ├── pylaia_recognition.py
│       └── {pylaia_himanis,pylaia_home_alcar}/
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
- `initial_pointer`: start position accounting for `--line-offset`

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

**Two-column handling:** at `label_idx == left_column_count`, the text pointer is hard-reset to the `column_break_777` anchor word index. The last col-1 line is force-closed at that anchor too, so no words fall through the gap.

**Mid-word breaks:** when `text_pointer + consumed` lands on a `MidWordBreak`, the last word of the current line is split at the volpiano syllable boundary using `_split_word_at_syl_boundary()`. The right fragment is stored in `syllable_prefix` and prepended to the next line's word list.

**No-volpiano folio-start location (`locate_folio_start`):** when `flat_text.anchors` is empty (all chants lack volpiano), the code searches for which line best matches the first folio chant via NW, declares that as line index L*, and treats lines 0…L*-1 as pre-start (previous folio's bleeding continuation). Pre-start lines receive either continuation words (via NW), suffix alignment words (preceding folio's last chant tail), or empty strings.

**Mixed-line detection:** for the last pre-start line (index L*-1), the code checks whether the rightmost constituent(s) of that fused line contain any opening words of this folio's first chant. If so, it constructs per-constituent `constituent_overrides` that correctly split the line between "previous folio continuation" and "this folio start".

**Output:** `AllocationResult(manifest, flags, text_pointer_end, debug_lines, folio_start_line, constituent_overrides)`

### 5c. `_defuse_manifest()`

After NW allocation, the manifest has fused labels (`fused_0`, `fused_1`, …). `_defuse_manifest` distributes each fused line's word list back to its constituent node labels proportionally by pixel width. The last constituent receives any remainder words to avoid rounding loss.

### 5d. Folio state persistence

`FolioState` carries `remaining_words` (post-77 continuation) and `last_chant_sequence` across folio runs. Written to a JSON sidecar via `write_folio_state()`; read back via `read_folio_state()`. Used to chain pipeline runs across consecutive folios of the same manuscript.

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
    initial_pointer: int                # start index (after line_offset)
    continuation_words: list[str]       # post-77 words for next folio
    has_continuation: bool              # True when prev-folio words were prepended
    suffix_probe_words: list[str]       # preceding folio's last chant words
```

### Pipeline JSON output schema

```json
{
  "folio": "168v_masked",
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
          "source": "gt",            // "gt" | "fallback"
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

The `source` field is `"gt"` when the manifest has a non-empty truthy value for that line's label, and `"fallback"` when the line fell through to OCR-based word segmentation.

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

## 8. Tools and External Dependencies

| Dependency | What it does | Where used |
|---|---|---|
| `kraken` | BLLA line segmentation; Kraken HTR recognition | `KrakenSegmentation`, `KrakenRecognition` |
| `htrflow` | Collection/SegmentNode tree; Result objects; PipelineStep base | all steps |
| `biopython` | Needleman-Wunsch alignment via `Bio.Align.PairwiseAligner` | `nw_chant_allocator.py` |
| `volpiano-display-utilities` | Latin syllabification | `syllable_segmentation.py`, `nw_chant_allocator.py` |
| `platformdirs` | Cross-platform user data directory (transitive dep of htrmopo) | `run_pipeline.py` |
| `numpy` | 1D coverage profile array; convolution smoothing | `column_clustering.py` |
| `opencv-python (cv2)` | BGR↔RGB conversion for PIL/Kraken | `KrakenSegmentation`, `KrakenRecognition` |
| `Pillow (PIL)` | Image format for Kraken; also used by `page_viewer.py` | multiple |
| `htrmopo` | Model download/caching (used via CLI only) | CLI workflow |

**Standard library:** `argparse`, `csv`, `io`, `json`, `logging`, `pathlib`, `re`, `statistics`, `unicodedata`, `urllib.request`.

**GUI deps (npm):** React, TypeScript, Vite, Tailwind CSS, OpenSeadragon, `@types/openseadragon`.

---

## 9. Notable Design Decisions

### HTRflow as scaffolding, not as the core

HTRflow provides the `Collection` tree and step protocol but contributes zero domain logic. All of the chant-specific behavior — co-linear fusion, NW alignment, syllabification, GT word segmentation — is implemented in `steps/`. HTRflow could be replaced with a different orchestration layer without touching the domain logic.

### NW alignment over direct volpiano line-splitting

The earlier approach (`build_page_manifest()` in `gt_manifest.py`) split chant text directly by volpiano `7` markers and zipped the result with node labels. This fails when chants start mid-line (previous chant ends partway through a physical line) — the zip assumes a clean one-to-one correspondence. The NW allocator avoids this assumption entirely: it uses OCR text as evidence to determine where each line starts and ends in the flat word sequence.

### Fused-line NW, constituent-label manifest

NW alignment runs on fused lines (one logical text line = potentially multiple BLLA segments). The fused-line OCR text is the concatenation of constituent OCR texts. After allocation, `_defuse_manifest` distributes words back to constituent labels proportionally by pixel width. This means the fused grouping is transparent to all downstream steps (Stages 5 and 6 operate on original node labels).

### Source field truthiness, not key presence

`_build_pipeline_payload` marks a word as `source="gt"` using `manifest.get(line_node.label)` (falsy check), not `line_node.label in manifest` (key existence). These diverge for lines that received an empty-string GT assignment (e.g. a line with no text in the Cantus CSV). `GroundTruthWordSegmentation` also uses `if not gt_text:` to trigger fallback. The two checks must agree — an empty-string assignment means no GT text was available, so the word should be `"fallback"`.

### Apple Silicon stub pattern

Every step that inherits from `htrflow.pipeline.steps` wraps the import in `try/except ImportError` and defines a local stub. This is not defensive programming for general failure — it is specifically for the Apple Silicon mmcv crash. The stubs are load-time shims, not runtime fallbacks. They should never be removed until HTRflow stops importing mmcv at module level.

---

## 10. Known Limitations and Future Work

| Area | Issue | Path forward |
|---|---|---|
| HTR model | Tridis was trained on baseline crops; we feed bbox crops → "severely degraded performance" warning | Train a chant-specific HTR model on bbox crops |
| Segmentation | BLLA over-segments on neumes; co-linear fusion is a post-hoc fix | Fine-tune BLLA on chant manuscripts (Kraken BLLA fine-tuning setup exists in the repo) |
| NW with no volpiano | Folio-start location is approximate; works at line granularity | No clear path; volpiano is the ground-truth structural signal |
| Multi-folio runs | Requires manually chaining `--folio-state-out` / `--prev-folio-state` | Could be automated in a batch runner |
| Word geometry | Character-proportional bbox distribution ignores actual character widths | Would require OCR character-level bounding boxes from Kraken (possible but not currently extracted) |
| GUI | Deployed build is static; updating requires `npm run build` + push | Already automated via GitHub Actions for `gui/` changes |
