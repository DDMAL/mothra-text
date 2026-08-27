# Pipeline Steps

This package implements the custom HTRflow pipeline steps used by `run_pipeline.py`.
Together they form a proof-of-concept pipeline for word-level ground-truth alignment
on medieval chant manuscripts.

## Steps overview

```
KrakenSegmentation  →  cluster_columns
                    →  KrakenRecognition
                    →  [pre-NW music filter]     # drops lines overlapping music regions
                    →  [pre-NW off-area filter]  # drops lines outside the main text area
                    →  fuse_colinear_segments
                    →  allocate_lines (NW chant allocator)
                    →  GroundTruthWordSegmentation
                    →  SyllableSegmentation
```

The pre-NW music filter is not a separate step class — it runs inline in `run_pipeline.run()`
when `music_boxes` is provided. See `run_pipeline.py` and the root `README.md` (§1c) for details.

The pre-NW off-area filter is likewise inline in `run_pipeline.run()`, gated on
`drop_offarea_boxes` (default `True`). Unlike the music filter it needs no external
annotation — `_main_text_area` derives the main chant text area per column from BLLA's own
line geometry, grouping lines into rows by y-overlap (the same rule `fuse_colinear_segments`
uses) so a real line fragmented into several small boxes is judged by its rows' combined
width, not any single box's. See the root `README.md` (§1e) for the user-facing description
and mothra-text#53 for the design rationale, including two approaches that were tried and
rejected before this one (fixed-distance padding, and extending the area via left-edge
alignment) because they either missed real marginalia or put genuine short lines at risk.

---

## `column_clustering.py`

### `cluster_columns(line_nodes, page_width, bimodal_threshold=0.5, min_gutter_fraction=0.02, min_peak_count=2, min_column_fraction=0.15, forced_column_count=None)`

Auto-detects 1 vs 2 columns using a **horizontal coverage-profile bimodal test**.
For each pixel column `x`, `coverage[x]` counts the number of line bounding boxes
that include that position.  A genuine 2-column page has two peaks in the coverage
profile (one per column) separated by a valley (the inter-column gutter); a 1-column
page has a single continuous plateau.

Returns `(sorted_labels, column_count, split_x)` where `sorted_labels` is the list
of node labels in reading order (left column top-to-bottom, then right),
`column_count` is 1 or 2, and `split_x` is the column-split x-coordinate (`None`
for single-column pages).

**Algorithm:**
1. Build the coverage array and smooth with a 5-px window.
2. Scope the search to the inner 20–80 % of the **text region**
   (`min(xmin) → max(xmax)`) to exclude blank page margins that would otherwise
   produce spurious zero-coverage gaps.
3. Find the deepest valley in the search band.
4. Compute `left_peak` and `right_peak` on each side of the valley.
5. Declare two columns when all of the following hold:
   - `valley < bimodal_threshold × min(left_peak, right_peak)` (default 0.5 — valley
     must be less than half the smaller peak),
   - both peaks reach `min_peak_count` (default 2 lines),
   - the gutter width ≥ `min_gutter_fraction × page_width` (default 2 %), and
   - each side of the candidate split contains at least `min_column_fraction` (default
     15 %) of all line nodes — rejects false splits driven by a handful of outlier
     segments (initials, neumes) far from the main text block.

BLLA spanning-bbox artefacts (one bbox drawn across both columns at the same
y-position) only slightly elevate the valley and do not eliminate the bimodal
structure provided they are a minority of the total lines.  The text-region scoping
means blank page margins never produce false two-column splits.

Use `--column-bimodal-threshold` to tune sensitivity; raise it to accept a shallower
valley (more aggressive 2-column detection), lower it to require a deeper valley
(stricter).

**`forced_column_count` parameter:** Pass `1` or `2` to bypass bimodal auto-detection
entirely. `forced_column_count=1` short-circuits before the coverage analysis and
returns nodes sorted by `ymin`. `forced_column_count=2` still runs the coverage
analysis to find `split_x`; if the bimodal test fails (e.g. a poorly-scanned gutter),
`split_x` falls back to the coverage-profile valley position, or to the page midpoint
as a last resort. Set via `--column-count` in `run_pipeline.py`.

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

### `KrakenRecognition(model=None, device="cpu", allow_stub=False)`

HTRflow pipeline step for per-line text recognition using a Kraken HTR model.

- **Default model**: `run_pipeline.py` automatically uses the Tridis model
  (`Tridis_Medieval_EarlyModern.mlmodel`) if it is installed via htrmopo. Install with:
  `python -m htrmopo get 10.5281/zenodo.10788591`. If no model is found and `--stub-mode`
  was not given, the CLI exits with an error.
- **Stub mode** (`model=None, allow_stub=True`): logs a WARNING and sets empty text on
  all line nodes. The pipeline still runs to completion, allowing segmentation and JSON
  export to be tested without a model. Must be triggered explicitly with `--stub-mode`.
  Passing `model=None` without `allow_stub=True` raises a `ValueError`.
- **Custom model** (`--recognition-model PATH`): loads any Kraken-compatible model
  (local `.mlmodel` path or HuggingFace model ID). Runs `kraken.rpred.rpred` on each
  line crop. Recognized text and per-character confidence scores are stored on the node.

Kraken imports are lazy (inside `run()`), so stub mode never requires Kraken to be
importable.

---

## `kraken_segmentation.py`

### `KrakenSegmentation(device="cpu", model=None)`

HTRflow pipeline step wrapping Kraken's BLLA baseline segmenter. Produces line-level
polygon nodes from a folio image.

- **Default mode** (`model=None`): uses Kraken's built-in BLLA model.
- **Custom model** (`model="path/to/model"`): accepts a local path to a `.mlmodel`
  (CoreML) or `.safetensors` file. The model is loaded once at construction time and
  reused across all pages in the collection.

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

### `build_flat_text_and_anchors(csv_rows, folio, prev_folio_state=None, infer_continuation=True)`

Builds `FlatTextData` from a Cantus CSV:

1. Filters and sorts rows for the target folio (excludes `mode="*"` rows).
2. Parses each row's volpiano for break markers:
   - `7` → `within_chant_7` anchor (line break within folio)
   - `77` → `page_break_77` anchor + truncation (post-77 words go to next folio via
     `continuation_words`)
   - `777` → `column_break_777` anchor
3. When a `prev_folio_state` IS given, it is authoritative even if its
   `remaining_words` is empty (the previous folio's chant genuinely terminated
   there, `fully_consumed=True`) — `infer_continuation` is never consulted in
   that case. An explicit "nothing to carry over" from the actual run is real
   information, not an absence of it; falling through to the CSV-guess below
   just because the list happens to be empty would silently overwrite that
   answer with a stale one (this is exactly what happened when a phantom
   folio like "003r" — fully consumed, correctly leaving nothing over — was
   followed by a real folio that still picked up an earlier folio's leftover
   continuation via the CSV-scan below).
4. Only when no `prev_folio_state` is given at all, and `infer_continuation=True`
   (default), automatically finds the nearest preceding folio by CSV ordering (not all
   preceding history — an intervening folio with no `77` would otherwise let a stale
   break from much earlier get misattributed) and, if its last row's volpiano has `77`,
   prepends the words belonging to the target's own physical position as a virtual
   ChantSpan (sequence 0). This means a single standalone folio run does not need to be
   chained to recover a mid-chant page boundary.

   The CSV-visible predecessor is not always the physically preceding folio — a folio
   with no chant starting on it gets no CSV row at all (mothra-text#42), so it is
   invisible to this scan. `build_flat_text_and_anchors` accounts for that gap: it
   counts how many physical folio-sides separate the CSV-visible predecessor from the
   target (via `_folio_linear_index`, which treats consecutive recto/verso sides as
   consecutive integers regardless of CSV presence), and uses `_carry_words_for_gap` to
   select the matching `page_break_77`-delimited segment of the predecessor's row —
   segment 0 for a true immediate predecessor, segment 1 when one row-less folio sits in
   between, and so on. A row normally has at most one `77`, but when a row-less folio's
   own text is embedded inside a neighbour's row, a *second* `77` marks where that
   folio's text ends and the next folio's begins (mothra-text#55). When the row does not
   have enough `77`s to reach as far as the target, no continuation is added — the
   row-less folio(s) in between absorbed everything, or there genuinely is none.

   Not every extra `77` in a row is a real folio boundary, though: some rows end
   `...<word>77---4` with nothing volpiano-encoded after the break — often a doxology
   whose routine cadence was never re-notated, even though the raw fulltext has a few
   more (untranscribed) words past it. `_parse_row_structure` drops that kind of
   trailing, content-free `77` rather than splitting it into its own segment, so those
   trailing words stay attached to whichever real break precedes them instead of being
   misattributed to a folio further away. Confirmed against real data: CH-Fco Ms. 2's
   `002v` row has exactly this shape (a genuine `77` into `003r`, then a dangling `77`
   right before the doxology's "et in secula seculorum amen") — the manuscript images
   show that whole phrase is still on `003r`, not `003v`.

   **Caveat:** this only resolves gaps caused by CSV-row-less folios, using evidence
   from the row's own break structure. It has no notion of "which folios a given run
   included" at all — it always operates on the CSV's own physical/row adjacency,
   regardless of whether the folio actually run before the target was its true
   predecessor. That is what makes it safe for callers to rely on unconditionally: a
   run that skips a folio which *does* have its own CSV row still resolves correctly
   (the scan finds that row directly), and a skipped CSV-row-less folio still yields no
   continuation rather than a stale one, exactly as described above. `run_chain.py`
   only resets `prev_folio_state` on a failed contiguity check and otherwise leaves
   `infer_continuation=True`, falling back to this scan the same way a standalone run
   of that folio would (mothra-text#58).
5. Cantus `|` phrase separators (e.g. an antiphon and its verse combined in one row)
   are their own whitespace-delimited token in `fulltext_ms`/`fulltext_standardized`
   and are stripped entirely by `clean_text()`, but the row's `volpiano` field still
   allocates a word-group position for them. `_parse_row_words_and_anchors` corrects
   every anchor/mid-word-break index for this offset (by counting `|` tokens in the
   raw, uncleaned text) before returning, so `mid_word_breaks`/anchors are always
   valid indices into the returned (already pipe-stripped) word list.

### `allocate_lines(flat_text, sorted_labels, ocr_texts, column_count=1, ...)`

Assigns a word fragment from `flat_text` to each line label via NW alignment:

- Uses `Bio.Align.PairwiseAligner` (requires `biopython`) with affine gap penalties
  calibrated for medieval chant (match 8, mismatch −5, gap open −7, gap extend −3).
- Advances a `text_pointer` through `flat_text.words` as each line is processed.
- Snaps the pointer to the nearest `within_chant_7` or `page_break_77` anchor when NW
  result is within `snap_window` words of it (default 2); emits `nw_volpiano_disagreement`
  flags when the gap exceeds the snap window.
- **Force-window snap** (`force_window`, default 10): when NW under-consumes due to poor
  OCR, forces the pointer to a `within_chant_7` anchor even when the gap exceeds
  `snap_window`, provided no new chant span starts between the current pointer and the
  anchor (i.e. the line is safely mid-chant). Emits `forced_mid_chant_snap` flags.
  Set `force_window=0` to disable entirely.
- Hard-resets the pointer to the nearest `column_break_777` anchor at the start of
  column 2 (when `column_count=2`).
- **Stub mode** (when OCR texts are all empty, i.e. `--stub-mode` was given or no
  recognition model was available): each line advances to the next anchor of any type
  instead of running NW.
  When no anchor is available (e.g. all chants on the folio lack volpiano), remaining
  words are distributed uniformly across remaining lines — `floor(remaining_words /
  remaining_lines)` per line, with the last line receiving any leftover words.
- **Misdetected-line skipping** (`skip_misdetected_lines=True`, default): BLLA sometimes
  draws a small box around a non-chant area — a neume group, a clef sliver, an initial.
  OCR correctly reads nothing there, which sends the box down the stub-mode branch above,
  where it swallows a whole line's worth of words and shifts every following line. A box is
  judged a non-text detection, assigned `""` and skipped (leaving the pointer where it was,
  so its words go to the next line) only when **all four** signals agree:
  1. allocation wants at least `misdetect_min_words` words from it (default 1 — this only
     exempts zero-demand no-ops);
  2. its OCR holds fewer than `misdetect_min_ocr_chars` **alphanumeric** characters
     (default 2, so punctuation-only output like `'„'` counts as nothing read);
  3. it is narrower than `misdetect_width_ratio` of the page's typical text-line width
     (default 0.35);
  4. its estimated character capacity is below `misdetect_capacity_ratio` of the text
     being assigned to it (default 0.4) — it cannot physically hold nearly that much.

  Conditions 3 and 4 are measured against page statistics from `_page_line_scale()`:
  the median width of "reference lines" (fused lines whose OCR has ≥ 8 alphanumeric
  characters) and the *narrowest* average character width among them, which is the most
  generous capacity estimate. Fewer than 3 reference lines means no trustworthy page scale
  and the feature stays inert — which also covers global stub mode, where no line qualifies.
  Requires `fused_lines`; without it the feature is inert.

  Emits `misdetected_line_skipped` (detail includes the bbox, the width as a percentage of
  typical, the word/character demand and the estimated capacity) and lists the labels in
  `AllocationResult.skipped_labels`. Applies in the main allocation loop **and** in both
  pre-start sub-branches, where a vetoed box is also exempted from the force-snap.
  Note that a genuinely short line whose OCR merely failed is protected by condition 4: its
  demand fits its width, so it is never skipped.

  **Threshold calibration.** Measured over 77 fused lines on 6 folios from 3 manuscripts
  (CH-Fco Ms. 2 002r/002v/006r, NZ-Wt MSR-03 002v, MS234 063v/064r, spanning 795–4872 px
  page widths), every box the rule actually skipped was ≤ 0.069 of its page's reference
  width, and the narrowest genuine text line was 0.376 — a `cula seculorum amen` closing
  line. The `misdetect_width_ratio` default of 0.35 therefore sits inside an empty band.
  Do not raise it: that closing line is only 7 % above it, and while it is independently
  protected by conditions 2 and 4 (17 alphanumeric OCR characters, and 25 characters of
  capacity against an 8.4 threshold), the margin on width alone is thin. Lowering it costs
  reach on small-format pages, where a non-text box is a larger fraction of a line: NZ-Wt
  MSR-03's reference width is only 467 px.

  Five of those folios were run unmasked; NZ-Wt MSR-03 002v was run **with** masking and
  still contained a 21 px non-text box that consumed 6 words — masking is itself a detection
  step and leaks, so this rule stays on for masked runs too. Re-running CH-Fco Ms. 2 002r
  with `--mothra-json` does yield 13 fused lines instead of 15, no non-text boxes and zero
  veto candidates: that is the negative control for these thresholds — on a page with nothing
  to catch, the rule goes fully inert.
- Emits `continuation_missing` when the first word of `flat_text` is lowercase and
  `flat_text.has_continuation` is False. The flag detail now also mentions that absent
  volpiano on the preceding folio prevents automatic inference and instructs the user
  to chain runs via `--folio-state-out` / `--prev-folio-state`.
- **No-volpiano folio-start location** (`locate_folio_start=True`, default): when
  `flat_text.anchors` is empty (no volpiano on any chant row) and at least one OCR
  line is non-empty, runs `locate_first_chant_line()` to find the OCR line whose text
  best matches the opening words of the first folio chant (the first `ChantSpan` with
  `sequence > 0`). Lines before that line (L\*) are treated as the previous folio's
  bleeding continuation:
  - If `flat_text.has_continuation` is True, NW-aligns the continuation words against
    the pre-start lines, then force-snaps the last pre-start line to consume any
    remaining continuation words (mirrors the `column_break_777` force-close pattern).
  - If `flat_text.has_continuation` is False, the **pre-start suffix alignment** runs
    (see below); if it is disabled or scores below threshold, pre-start lines receive
    empty manifest entries.
  - The pointer is hard-reset to `first_folio_span.start_word` at L\* before the
    normal folio NW pass begins.
  - Emits `folio_start_detected` (informational, detail includes L\* and score) when
    L\* > 0, or `folio_start_not_located` (fallback to line 0) when every OCR line
    scores below `folio_start_min_score` (default 0.0).
  - Set `locate_folio_start=False` to disable, or raise `folio_start_min_score` to
    require a stronger match.  `folio_start_n_probe` (default 8) controls how many
    words of the first folio chant are used as the probe string.
  - Works at line granularity; if the first chant begins mid-line, L\* points to
    that mixed line (known approximation).
- **Mixed-line detection** (`fused_lines`, `node_ocr`, `mixed_line_n_words=3`,
  `mixed_line_min_score=0.0`): when `folio_start_line > 0` and no volpiano anchors
  are present, checks whether the rightmost constituents of fused line L\*−1 contain
  opening words of this folio's first chant. A grid search over (split index k,
  word count n ≤ `mixed_line_n_words`) scores per-constituent OCR against the first
  N folio words using NW; the best-scoring (k, n) pair is selected when its
  normalised score meets `mixed_line_min_score`. On success, `constituent_overrides`
  in `AllocationResult` maps each constituent of L\*−1 to its final text (left
  portion → pre-start text, right portion → first N folio words), and the hard-reset
  pointer at L\* is offset by N to avoid repeating those words. Emits
  `mixed_start_detected` (flag detail includes word count, constituent index, and
  score). Requires `fused_lines` and `node_ocr` to be supplied; silently skips
  otherwise.
- **Pre-start suffix alignment** (`pre_start_suffix_align=True`, default): when
  `has_continuation=False` and `folio_start_line > 0` (pre-start lines were found),
  assigns CSV ground-truth text to those pre-start lines by aligning their
  concatenated OCR against the preceding folio's last chant text. The preceding
  folio's last chant is stored in `FlatTextData.suffix_probe_words` (populated by
  `build_flat_text_and_anchors` when no `77` continuation was found). A semi-global
  NW alignment with free left gaps on the target finds the split index `k*` — the
  word in the preceding chant where this folio's bleed-over begins — and then
  distributes `chant_words[k*:]` across the pre-start lines using forward NW, with
  force-snap on the last pre-start line. Emits `suffix_alignment_detected` (with
  word offset and score) on success, or `suffix_alignment_skipped` when the score
  is below `pre_start_suffix_min_score` (default 0.0), in which case pre-start
  lines fall back to empty manifest entries. Set `pre_start_suffix_align=False` to
  disable entirely.

### `build_folio_state / write_folio_state / read_folio_state`

Builds, serialises, and deserialises a `FolioState` JSON sidecar after each folio run.
`remaining_words` captures post-77 continuation words (or the unconsumed flat_text
tail) for use as `prev_folio_state` on the next folio run.

---

## `syllable_segmentation.py`

### `SyllableSegmentation()`

HTRflow pipeline step that subdivides each word-level node into character-proportional
syllable regions. Operates after `GroundTruthWordSegmentation` on the word-level active
leaves.

- Normalises word text to plain lowercase ASCII (NFKD decomposition + ASCII
  transliteration) before syllabification to handle accented vowels and ligatures
  (`æ`, `œ`, etc.) from Cantus `fulltext_ms` fields.
- Delegates to `syllabify_word` / `split_word_by_syl_bounds` from
  `volpiano-display-utilities`. Non-final syllables carry a trailing hyphen
  (e.g. `["do-", "mi-", "nus"]`).
- Falls back to treating the whole word as a single syllable on `LatinError` or
  empty normalisation; always produces at least one child per word node.
- Bounding boxes are laid out character-proportionally within the word node's
  coordinate space.

Requires `volpiano-display-utilities` (`pip install volpiano-display-utilities`).

---

## `mothra_mask.py`

### `MothraImageMask(mothra_json_path, padding_px=25)`

Produces a masked version of a folio image that shows only mothra classId-1 (text)
regions; all other pixels are set to black. Invoked by `run_pipeline.run()` when
`mothra_json_path` is provided (either via the `--mothra-json` CLI flag or passed
directly by a library caller such as the landing page backend).

- **`apply(pil_image) → PIL.Image`** — for each classId-1 bbox `[x, y, w, h]` in the
  mothra JSON, paints a white rectangle expanded by `padding_px` on all sides (clamped
  to image bounds) on a black canvas, then composites the original image through the
  mask. The default `padding_px=15` merges adjacent word-level detections on the same
  physical line into a continuous strip that Kraken can detect as a full line.

---

## `gt_manifest.py`

Builds the `gt_lookup` callable from a Cantus CSV export.

- `fetch_cantus_csv(source_id)` — downloads from `cantusdatabase.org/source/{id}/csv/`
- `load_local_csv(path)` — reads a local Cantus-format CSV
- `make_manifest_lookup(manifest)` — wraps a `{node_label: text}` dict as a callable
- `clean_text(text)` — strips Cantus `|` phrase separators and collapses whitespace. See
  `nw_chant_allocator.py`'s `build_flat_text_and_anchors` point 6 above for how its callers
  correct volpiano-derived indices for the word position this removes.

---

## `ground_truth_word_segmentation.py`

### `GroundTruthWordSegmentation(gt_lookup)`

Drop-in replacement for HTRflow's `WordSegmentation` step. For each line node, looks up
its text in `gt_lookup` and lays out word bounding boxes using a uniform
pixels-per-character formula. Falls back to recognition-based segmentation and logs a
warning when `gt_lookup` returns `None`.

Each resulting word segment is stamped with `source` (`"gt"` or `"fallback"`) at build
time, when the GT-vs-fallback decision is actually known. `_build_pipeline_payload`
reads this tag directly off the word node (`word_node.get("source", "fallback")`)
rather than re-deriving it later from `manifest.get(line_node.label)` — line labels can
shift after this step's own `collection.update()` call triggers HTRflow's
`Collection.relabel()`, which would desync a label-based lookup from the manifest it
was built against (mothra-text#59).

---

## Running the tests

```bash
conda activate line-seg-eval
pytest tests/ -v
```

400 tests across `test_column_clustering.py`, `test_nw_flat_text.py`,
`test_nw_alignment.py`, `test_nw_folio_state.py`, `test_run_pipeline.py`, and others.

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
  `build_flat_text_and_anchors` emits a `logger.warning` summary when any chant rows
  on the folio lack volpiano (e.g. "3 of 7 chant row(s) on folio '006r' have no
  volpiano"). In stub mode, lines without anchors now receive a uniform share of
  remaining words instead of the previous 1-word-per-line fallback.
- **Misdetected-line skipping is Cantus-mode only.** OCR-only mode never calls
  `allocate_lines`: each line is segmented from its own OCR by
  `_fallback_word_segmentation`, so there is no shared text pointer for a non-text box to
  corrupt and nothing to veto.
- **Misdetected-line skipping and column 1.** When a vetoed box is the last line of the
  left column, the words in the gap fall to the `column_break_777` hard-reset rather than
  being allocated, since the box is deliberately exempted from the column force-close.
- **Misdetected-line skipping cannot catch plausibly-sized non-text boxes.** A box wide
  enough to hold the word it is offered passes condition 4 whatever it actually contains —
  e.g. a 379px neume group offered `in`, or a 102px margin annotation offered `Ne`. A box
  lying outside the main chant text area entirely (folio numbers, running heads, marginal
  notes) is caught instead by the separate pre-fusion off-area filter (`_main_text_area`,
  `run_pipeline.py`; root `README.md` §1e) — a box's y-centre sitting off the page's
  regular text-line pitch, but still inside the block's x/y footprint, remains uncaught by
  either mechanism.
- **The off-area filter only catches marginalia outside the block's true footprint.** A
  box that sits *within* the main chant text area's x/y extent — an interlinear rubric cue
  sandwiched between two real lines, for instance — is geometrically indistinguishable from
  a genuine short line and is deliberately left alone. On CH-Fco Ms. 2 009v this describes
  most of the marginal rubric boxes on the page (see mothra-text#52's 2026-08-27 comment);
  telling them apart from real text needs region classification, not geometry.
