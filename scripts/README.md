# Scripts

## mothra_to_page.py

Converts Mothra Annotator JSON exports to PAGE XML (2019-07-15) for Kraken BLLA training.

Keeps only `classId 1` (chant text lines) and discards column-level bounding boxes above a configurable height threshold.

```
python scripts/mothra_to_page.py <input_dir> [options]

positional arguments:
  input_dir             directory of .json files to convert

options:
  --output-dir PATH     destination for .xml files
                        (default: <input_dir>/page_xml/)
  --height-filter FRAC  discard annotations taller than this fraction of
                        image height (default: 0.15)
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR (default: INFO)
```

**Example**

```
python scripts/mothra_to_page.py \
    ~/Downloads/DDMAL/Annotation/MtCassinoCod127_0008_annotations/ \
    --output-dir data/page_xml/
```

One `.xml` is written per `.json`; failures are logged and the batch continues.

---

## build_gt_manifest.py

Builds a ground-truth manifest for one folio page. Downloads the Cantus CSV for a manuscript source, splits each chant's text into per-line fragments using volpiano `7` markers, and writes a JSON manifest mapping HTRflow node labels to their Cantus text fragments.

```
python scripts/build_gt_manifest.py \
    --source-id <int> \
    --folio <str> \
    --node-labels-file <path> \
    --output <path>
```

The `--source-id` is the integer Cantus source ID from cantusdatabase.org. The node-labels file should contain one HTRflow node label per line in reading order (readable from `id` attributes on `<TextLine>` elements in PAGE XML output).

**Example**

```
python scripts/build_gt_manifest.py \
    --source-id 123723 \
    --folio "006r" \
    --node-labels-file node_labels.txt \
    --output data/manifests/manifest_006r.json
```
