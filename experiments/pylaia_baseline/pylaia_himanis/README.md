# PyLaia Himanis Baseline

Zero-shot CER baseline on 4 Gothic textualis chant manuscript folios using
the pre-trained model [`Teklia/pylaia-himanis`](https://huggingface.co/Teklia/pylaia-himanis),
before any fine-tuning. Parallel to the [home-alcar baseline](../pylaia_home_alcar/README.md)
for direct model comparison.

## Folios

Defined in `../folios.txt` (shared across all pylaia_baseline experiments):

| File | Source |
|---|---|
| `AM_20_b_I_fol.-15-0006v-hq.pdf` | Arnamagnæan Institute |
| `Antiphonal_12v_hfngl.jpg` | — |
| `BeromunsterCantatorium_10v.jpg` | Beromünster Cantatorium |
| `CH-Fco Ms. 2_006r copy.jpg` | Fribourg, Cordeliers |

## Pipeline

```
01_segment.py      Kraken BLLA → line coordinate JSON + visualisation  [shared]
02_extract_crops.py JSON → 128px-high grayscale line crop PNGs         [shared]
03_run_pylaia.py   crops → transcriptions via pylaia-htr-decode-ctc
```

Stages 1 and 2 are shared scripts at the `pylaia_baseline/` level and write
to `outputs/pylaia_baseline/segmentation/` and `outputs/pylaia_baseline/crops/`.
Both sub-experiments read from those shared directories; stage 3 is the only
stage that writes new outputs.

Run all stages at once:
```bash
conda activate line-seg-eval
python experiments/pylaia_baseline/pylaia_himanis/run_experiment.py
```

Or step by step from the repo root:
```bash
python experiments/pylaia_baseline/01_segment.py
python experiments/pylaia_baseline/02_extract_crops.py
python experiments/pylaia_baseline/pylaia_himanis/03_run_pylaia.py
```

All stages are idempotent — re-running skips already-completed work.

## Outputs

```
outputs/pylaia_baseline/
├── segmentation/
│   └── {stem}.json               Kraken BLLA line coords (shared)
├── crops/
│   └── {stem}/
│       └── line_{id:04d}.png     128px-high grayscale line crops (shared)
└── pylaia_himanis/
    ├── transcriptions/
    │   └── {stem}.txt            Raw pylaia-htr-decode-ctc output per folio
    └── results.csv               folio, line_id, transcription (all lines)
```

## Environment setup

Segmentation and crop extraction run in the main `line-seg-eval` env.
PyLaia inference requires a **separate** `pylaia-env` because PyLaia 1.1.x
depends on torch 1.13 while the rest of the repo requires torch 2.x.

```bash
# Main env (if not already set up — see top-level README)
conda create -n line-seg-eval python=3.10 -y
conda activate line-seg-eval
pip install htrflow kraken pymupdf huggingface_hub

# PyLaia env (one-time setup)
conda create -n pylaia-env python=3.10 -y
conda activate pylaia-env
pip install pylaia huggingface_hub "setuptools<72" "torchmetrics==0.4.1"
```

`03_run_pylaia.py` calls `pylaia-htr-decode-ctc` directly from
`~/miniconda3/envs/pylaia-env/bin/` so you do not need to activate
`pylaia-env` manually.

The `Teklia/pylaia-himanis` model weights are downloaded automatically
on first run via HuggingFace Hub and cached in `~/.cache/huggingface/`.
The model architecture file is cached separately in `~/.cache/pylaia-himanis/`.

## Notes

- **No ground truth is included.** `results.csv` contains raw transcriptions
  for manual review against the folio images. Computing actual CER requires
  a reference transcription layer, which is the next step in the pipeline.
- The model was trained on HIMANIS (French medieval documents) and HOME-Alcar
  (Latin medieval documents). Zero-shot performance on Gothic textualis chant
  material is expected to be imperfect — that's the point.
- YOLO returned 0 lines on some folios; Kraken BLLA is used here because it
  reliably produces boundary polygons needed for crop extraction.
- The Himanis checkpoint uses the same CRNN architecture as `pylaia-home-alcar`
  (`[12,24,48,48]` features, `[[2,1],...]` poolsize, batchnorm on, 3 RNN layers),
  confirmed by inspecting checkpoint state_dict shapes on first run.
