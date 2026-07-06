# Scripts

## run_mothra_inference.py

Runs the DDMAL-lab YOLOv11 mothra models on folio images to produce annotation JSONs
compatible with `run_pipeline_mothra.py`. Downloads `text_music_detector_fulldata.pt`
and `stave_detector_fulldata.pt` from `DDMAL-lab/mothra-yolov11-checkpoints` on first
run (cached by `huggingface_hub`). Requires HF token at `~/.cache/huggingface/token`.

Output schema matches the mothra Annotator export format (classId 1=text, 2=music,
3=staves; bbox in absolute pixels `[x, y, w, h]`).

```bash
python scripts/run_mothra_inference.py \
    --images path/to/folio1.jpg path/to/folio2.jpg \
    --out-dir ~/Downloads/DDMAL/mothra-text-layer/JSONs/Uncorrected/
```

Use `--conf` to adjust the YOLO confidence threshold (default 0.25).

---

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

## visualize_mothra.py

Overlays mothra object detection annotations on a folio image for visual inspection.

```
python scripts/visualize_mothra.py JSON_PATH IMAGE_PATH OUTPUT_PATH
```

Draws classId-1 (text) bboxes in green, classId-2 (music) in blue, and classId-3
(staves) in red. Saves the annotated image to `OUTPUT_PATH` and prints per-class counts.

**Example**

```
python scripts/visualize_mothra.py \
    ~/Downloads/DDMAL/mothra-text-layer/JSONs/Antiphonal_12v_hfngl.json \
    ~/Downloads/DDMAL/mothra-text-layer/images/Antiphonal_12v_hfngl.jpg \
    ~/Downloads/DDMAL/mothra_viz_12v.jpg
```

---

## compare_runs.py

Reads two or more pipeline export JSONs and prints a comparison table of line counts
and word-source statistics (GT vs. fallback). Useful for evaluating the effect of
mothra integration approaches against the baseline.

```
python scripts/compare_runs.py \
    --label baseline ~/Downloads/DDMAL/baseline_12v.json \
    --label masked   ~/Downloads/DDMAL/mothra_masked_12v.json \
    --label union    ~/Downloads/DDMAL/mothra_union_12v.json \
    --output ~/Downloads/DDMAL/mothra_comparison_report_YYYY-MM-DD.txt
```

The `--output` file is never overwritten; choose a new name each run.

---

## convert_to_mei_input.py

Converts a mothra-text `--export-json` output file to the **Text Alignment JSON** format
expected by the MEI encoding job.

```
python scripts/convert_to_mei_input.py --input PATH --output PATH [options]

required arguments:
  --input PATH          path to mothra-text pipeline JSON (from --export-json)
  --output PATH         destination path for MEI Text Alignment JSON

options:
  --exclude-fallback    skip syllables from lines that received no Cantus
                        ground truth text (source == "fallback")
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR (default: INFO)
```

**Example**

```
python scripts/convert_to_mei_input.py \
    --input ~/Downloads/DDMAL/006r_pipeline.json \
    --output ~/Downloads/DDMAL/006r_mei_input.json
```

The output JSON has two keys: `median_line_spacing` (75th-percentile of line-spacing
differences, matching the original Rodan text alignment formula) and `syl_boxes` (flat
list of `{syl, ul, lr}` entries in reading order).

