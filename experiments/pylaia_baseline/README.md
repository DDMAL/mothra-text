# PyLaia Baseline Experiment

Zero-shot CER baseline on 4 Gothic textualis chant manuscript folios using
the pre-trained model [`Teklia/pylaia-home-alcar`](https://huggingface.co/Teklia/pylaia-home-alcar),
before any fine-tuning. This establishes a rough starting point for a
medieval chant HTR pipeline.

## Folios

Defined in `folios.txt`:

| File | Source |
|---|---|
| `AM_20_b_I_fol.-15-0006v-hq.pdf` | Arnamagnæan Institute |
| `Antiphonal_12v_hfngl.jpg` | — |
| `BeromunsterCantatorium_10v.jpg` | Beromünster Cantatorium |
| `CH-Fco Ms. 2_006r copy.jpg` | Fribourg, Cordeliers |

## Pipeline

```
01_segment.py      Kraken BLLA → line coordinate JSON + visualisation
02_extract_crops.py JSON → 128px-high grayscale line crop PNGs
03_run_pylaia.py   crops → transcriptions via pylaia-htr-decode-ctc
```

Run all stages at once:
```bash
conda activate line-seg-eval
python experiments/pylaia_baseline/run_experiment.py
```

Or step by step from the repo root:
```bash
python experiments/pylaia_baseline/01_segment.py
python experiments/pylaia_baseline/02_extract_crops.py
python experiments/pylaia_baseline/03_run_pylaia.py
```

All stages are idempotent — re-running skips already-completed work.

## Outputs

```
outputs/pylaia_baseline/
├── segmentation/
│   └── {stem}.json               Kraken BLLA line coords (baseline + boundary)
├── crops/
│   └── {stem}/
│       └── line_{id:04d}.png     128px-high grayscale line crops
├── transcriptions/
│   └── {stem}.txt                Raw pylaia-htr-decode-ctc output per folio
└── results.csv                   folio, line_id, transcription (all lines)
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

The `Teklia/pylaia-home-alcar` model weights are downloaded automatically
on first run via HuggingFace Hub and cached in `~/.cache/huggingface/`.

## Notes

- **No ground truth is included.** `results.csv` contains raw transcriptions
  for manual review against the folio images. Computing actual CER requires
  a reference transcription layer, which is the next step in the pipeline.
- The model was trained on HOME-Alcar (Latin medieval documents) and Himanis
  (French medieval documents). Zero-shot performance on Gothic textualis chant
  material is expected to be imperfect — that's the point.
- YOLO returned 0 lines on some folios; Kraken BLLA is used here because it
  reliably produces boundary polygons needed for crop extraction.
