# Ground-Truth-Aware Word Segmentation

This package implements a proof-of-concept HTRflow pipeline step for word segmentation
driven by Cantus ground-truth text. In the broader HTR-OMR alignment pipeline, the goal
is syllable-level bounding boxes that can be paired with neume groups from the music
segmentation. Word segmentation is the intermediate step that makes syllable segmentation
possible. The problem with using the HTR recognised text for word counting is that CER
propagates directly into wrong bounding-box geometry: a misread word shifts every
subsequent box. By sourcing word identity and count from the authoritative Cantus text
instead — while keeping HTRflow's existing pixels-per-character geometry unchanged —
this step produces correct word boundaries for any line where Cantus data is available.

## How it works

**`GroundTruthWordSegmentation`** ([ground_truth_word_segmentation.py](ground_truth_word_segmentation.py))

- Drop-in replacement for HTRflow's `WordSegmentation` pipeline step.
- For each line node, calls `gt_lookup(node)` to retrieve its Cantus text fragment.
- Splits that text on whitespace to get word count and identity, then lays out
  word bounding boxes using the same pixels-per-character formula as HTRflow's default.
- When `gt_lookup` returns `None` (no Cantus data for that line), falls back to
  recognition-based segmentation and logs a warning. The results list always has
  exactly one entry per active leaf node, satisfying HTRflow's `collection.update()`
  constraint.

**`gt_manifest.py`** ([gt_manifest.py](gt_manifest.py))

- Builds the `gt_lookup` callable from a Cantus CSV export (no manual annotation needed).
- Downloads the CSV from `cantusdatabase.org/csv/{source_id}`.
- Uses the volpiano column's `7` line-break markers to split each chant's text into
  per-manuscript-line fragments: `7` = line break, `77` = page break, `777` = column
  break, `---` = word boundary. The number of `---`-separated groups between breaks
  gives the word count for that line.
- Zips the resulting ordered fragment list with the HTRflow node labels for the folio,
  producing a `{node_label: text_fragment}` manifest.

## Usage

End-to-end in Python:

```python
from steps.gt_manifest import fetch_cantus_csv, build_page_manifest, make_manifest_lookup
from steps.ground_truth_word_segmentation import GroundTruthWordSegmentation

csv_rows = fetch_cantus_csv(source_id)          # cantusdatabase.org/csv/{source_id}
manifest = build_page_manifest(csv_rows, folio="006r", node_labels=[...])
step = GroundTruthWordSegmentation(gt_lookup=make_manifest_lookup(manifest))
collection = step.run(collection)
```

`source_id` is the integer ID listed on the source's detail page on cantusdatabase.org.
`node_labels` is the list of HTRflow node labels for the folio's line nodes, read from
the `id` attributes of `<TextLine>` elements in HTRflow's PAGE XML output.

A pre-built manifest JSON can also be loaded from disk:

```python
from steps.gt_manifest import load_manifest, make_manifest_lookup

manifest = load_manifest("data/manifests/manifest_006r.json")
gt_lookup = make_manifest_lookup(manifest)
```

## Building a manifest from the CLI

```bash
python scripts/build_gt_manifest.py \
    --source-id 123723 \
    --folio "006r" \
    --node-labels-file node_labels.txt \
    --output data/manifests/manifest_006r.json
```

`node_labels.txt` should contain one HTRflow node label per line, in reading order.

## Running the tests

```bash
conda activate line-seg-eval
pytest tests/ -v
```

## Limitations

- **Geometry is still crude**: word boundaries are laid out with a uniform
  pixels-per-character ratio derived from the line width and total character count.
  The step fixes word *count* and *identity*; it does not improve spatial precision.
- **Abbreviation mismatch**: Cantus text is expanded (standardised spellings), while
  the manuscript images contain abbreviated text. The expanded GT text is longer than
  what appears on the page, which compresses the pixels-per-character ratio and shifts
  word boxes slightly to the left.
- **Volpiano coverage**: volpiano notation is absent for roughly 60–70% of Cantus
  chants. Lines without volpiano data fall back to HTRflow's recognition-based
  segmentation.
