# Pipeline Steps

This package implements the custom HTRflow pipeline steps used by `run_pipeline.py`.
Together they form a proof-of-concept pipeline for word-level ground-truth alignment
on medieval chant manuscripts.

## Steps overview

```
KrakenSegmentation  →  cluster_columns / fuse_colinear_segments
                    →  KrakenRecognition
                    →  allocate_lines (NW chant allocator)
                    →  GroundTruthWordSegmentation
```

---

## `column_clustering.py`

### `cluster_columns(line_nodes, page_width, variance_threshold=0.5)`

Auto-detects 1 vs 2 columns from line left-edge x-coordinates using two independent
signals (bimodal xmin variance ratio and disjoint horizontal extents). Returns
`(sorted_labels, column_count, split_x)` where `sorted_labels` is the list of node
labels in reading order (left column top-to-bottom, then right), `column_count` is
1 or 2, and `split_x` is the column-split x-coordinate (`None` for single-column pages).

### `FusedLine` dataclass

Represents one logical text line produced by fusing one or more BLLA sub-segments.
Fields: `label` (synthetic, e.g. `"fused_0"`), `constituent_labels` (original node
labels sorted left-to-right by xmin), `constituent_widths` (pixel widths for
proportional word distribution), merged `xmin/xmax/ymin/ymax`, `column` (1 or 2).

### `fuse_colinear_segments(line_nodes, split_x, overlap_threshold=0.5)`

Groups BLLA sub-segments that belong to the same physical text line. Kraken BLLA
frequently splits one text line into 2–4 segments when horizontal gaps between word
groups (caused by neume notation) are detected as baseline boundaries. This function
corrects that over-segmentation before NW alignment.

**Algorithm:** within each column, segments are processed top-to-bottom; a new
segment is merged into the current group if its y-extent overlaps the group's merged
y-extent by at least `overlap_threshold × min(segment_height, group_height)` (default
50%). Constituents within each fused group are sorted left-to-right by xmin.

The result reduces e.g. 24 BLLA segments to ~14 logical lines on a typical chant
folio, improving both reading order and NW word-count allocation.

---

## `kraken_recognition.py`

### `KrakenRecognition(model=None, device="cpu")`

HTRflow pipeline step for per-line text recognition using a Kraken HTR model.

- **Stub mode** (`model=None`): logs a WARNING and sets empty text on all line nodes.
  The pipeline still runs to completion, allowing segmentation and JSON export to be
  tested without a model.
- **Model mode**: loads any Kraken-compatible model (local `.mlmodel` path or
  HuggingFace ID via `kraken.lib.models.load_any`). Runs `kraken.rpred.rpred` on each
  line crop. Recognized text and per-character confidence scores are stored on the node.

Kraken imports are lazy (inside `run()`), so stub mode never requires Kraken to be
importable.

---

## `kraken_segmentation.py`

### `KrakenSegmentation(device="cpu")`

HTRflow pipeline step wrapping Kraken's BLLA baseline segmenter. Produces line-level
polygon nodes from a folio image. Uses Kraken's built-in default BLLA model.

---

## `nw_chant_allocator.py`

Aligns Cantus CSV chant text to physical folio lines using Needleman-Wunsch sequence
alignment. Replaces the earlier `build_page_manifest()` approach, which broke when a
chant started mid-line after the previous chant ended.

### Data structures

| Class | Purpose |
|---|---|
| `Anchor` | A word index where volpiano signals a line break (`within_chant_7`, `page_break_77`, `column_break_777`) |
| `ChantSpan` | Maps a CSV sequence number to a word-index range in `flat_text.words` |
| `FlatTextData` | All chant words concatenated in CSV sequence order, with anchors, spans, and continuation metadata |
| `AllocationResult` | NW output: `manifest` (label → word string), `flags`, `text_pointer_end` |
| `ValidationFlag` | Quality signal emitted when NW and volpiano disagree, line counts mismatch, etc. |
| `FolioState` | JSON-serialisable state capturing post-77 continuation words for the next folio |

### `build_flat_text_and_anchors(csv_rows, folio, line_offset=0, prev_folio_state=None, infer_continuation=True)`

Builds `FlatTextData` from a Cantus CSV:

1. Filters and sorts rows for the target folio (excludes `mode="*"` rows).
2. Parses each row's volpiano for break markers:
   - `7` → `within_chant_7` anchor (line break within folio)
   - `77` → `page_break_77` anchor + truncation (post-77 words go to next folio via
     `continuation_words`)
   - `777` → `column_break_777` anchor
3. When `infer_continuation=True` (default) and no `prev_folio_state` is given,
   automatically scans all CSV rows for the last row from any preceding folio with `77`
   in its volpiano and prepends the post-77 words as a virtual ChantSpan (sequence 0).
   This means folio runs do not need to be chained sequentially to handle mid-chant
   page boundaries.
4. `line_offset` skips the first N `within_chant_7` anchors, setting `initial_pointer`
   accordingly (for images that are crops starting partway through a folio).

### `allocate_lines(flat_text, sorted_labels, ocr_texts, column_count=1, ...)`

Assigns a word fragment from `flat_text` to each line label via NW alignment:

- Uses `Bio.Align.PairwiseAligner` with affine gap penalties calibrated for medieval
  chant (match 8, mismatch −5, gap open −7, gap extend −3).
- Advances a `text_pointer` through `flat_text.words` as each line is processed.
- Snaps the pointer to the nearest `within_chant_7` or `page_break_77` anchor when NW
  result is within `snap_window=1` word of it; emits `nw_volpiano_disagreement` flags
  when the gap exceeds the snap window.
- Hard-resets the pointer to the nearest `column_break_777` anchor at the start of
  column 2 (when `column_count=2`).
- Emits `continuation_missing` when the first word of `flat_text` is lowercase and
  `flat_text.has_continuation` is False.

### `build_folio_state / write_folio_state / read_folio_state`

Builds, serialises, and deserialises a `FolioState` JSON sidecar after each folio run.
`remaining_words` captures post-77 continuation words (or the unconsumed flat_text
tail) for use as `prev_folio_state` on the next folio run.

---

## `gt_manifest.py`

Builds the `gt_lookup` callable from a Cantus CSV export.

- `fetch_cantus_csv(source_id)` — downloads from `cantusdatabase.org/source/{id}/csv/`
- `load_local_csv(path)` — reads a local Cantus-format CSV
- `make_manifest_lookup(manifest)` — wraps a `{node_label: text}` dict as a callable

---

## `ground_truth_word_segmentation.py`

### `GroundTruthWordSegmentation(gt_lookup)`

Drop-in replacement for HTRflow's `WordSegmentation` step. For each line node, looks up
its text in `gt_lookup` and lays out word bounding boxes using a uniform
pixels-per-character formula. Falls back to recognition-based segmentation and logs a
warning when `gt_lookup` returns `None`.

---

## Running the tests

```bash
conda activate line-seg-eval
pytest tests/ -v
```

172 tests across `test_column_clustering.py`, `test_nw_flat_text.py`,
`test_nw_alignment.py`, `test_nw_folio_state.py`, and others.

---

## Limitations

- **OCR quality is the main bottleneck.** When the HTR model produces few or no tokens
  for a line, NW consumes correspondingly few flat_text words, causing downstream lines
  to receive the words that belonged on the poorly-recognised line. A model trained on
  the target manuscript family will significantly improve alignment quality.
- **Segment fusion threshold is fixed at 50%.** On manuscripts with very tight line
  spacing the threshold may need lowering; on manuscripts where BLLA is accurate it can
  be set to 1.0 to disable fusion entirely (`--overlap-threshold` is not yet a CLI flag).
- **Word geometry is still crude.** Word boundaries are laid out with a uniform
  pixels-per-character ratio. Spatial precision improves once a better segmentation
  model is trained.
- **Volpiano coverage.** Volpiano notation is absent for roughly 60–70% of Cantus
  chants. Lines without volpiano anchors rely on NW alone, with no snap behaviour.
