# PyLaia Baseline Experiments

Zero-shot HTR baselines on 4 Gothic textualis chant manuscript folios,
comparing pre-trained PyLaia models before any fine-tuning.

## Sub-experiments

| Model | Directory | Output |
|---|---|---|
| [`Teklia/pylaia-home-alcar`](https://huggingface.co/Teklia/pylaia-home-alcar) | [pylaia_home_alcar/](pylaia_home_alcar/README.md) | `outputs/pylaia_baseline/pylaia_home_alcar/` |
| [`Teklia/pylaia-himanis`](https://huggingface.co/Teklia/pylaia-himanis) | [pylaia_himanis/](pylaia_himanis/README.md) | `outputs/pylaia_baseline/pylaia_himanis/` |

## Shared infrastructure

`01_segment.py`, `02_extract_crops.py`, and `folios.txt` live here and are
shared across all sub-experiments. Both models use the same Kraken BLLA
segmentation and the same 128px-high grayscale line crops.

Shared outputs (written once, read by all sub-experiments):
```
outputs/pylaia_baseline/
├── segmentation/         Kraken BLLA line coordinate JSON (one file per folio)
├── crops/                128px-high grayscale line crop PNGs (one dir per folio)
├── pylaia_home_alcar/    transcriptions/ + results.csv
└── pylaia_himanis/       transcriptions/ + results.csv
```

## Folios

| File | Source |
|---|---|
| `AM_20_b_I_fol.-15-0006v-hq.pdf` | Arnamagnæan Institute |
| `Antiphonal_12v_hfngl.jpg` | — |
| `BeromunsterCantatorium_10v.jpg` | Beromünster Cantatorium |
| `CH-Fco Ms. 2_006r copy.jpg` | Fribourg, Cordeliers |

## Running

Each sub-experiment has its own `run_experiment.py` that calls the shared
stages 1–2 first (idempotent — they skip if outputs already exist) then runs
its own stage 3:

```bash
conda activate line-seg-eval

# home-alcar model
python experiments/pylaia_baseline/pylaia_home_alcar/run_experiment.py

# himanis model
python experiments/pylaia_baseline/pylaia_himanis/run_experiment.py
```

See each sub-experiment's README for environment setup and output details.
