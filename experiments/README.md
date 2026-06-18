# Experiments

Comparative and exploratory experiments. These are **not part of the main PoC pipeline** (`run_pipeline.py`) — they are historical research artifacts used to evaluate alternative approaches.

For the main pipeline, see the [root README](../README.md).

---

## 1. Line Segmentation Model Comparison

Compares three line detection models on the same folio images:

| Model | Type | Runner |
|---|---|---|
| Kraken BLLA | Baseline detection | [`../run_kraken.py`](../run_kraken.py) |
| YOLOv9 (`Riksarkivet/yolov9-lines-within-regions-1`) | Object detection | [`run_htrflow.py`](run_htrflow.py) |
| RTMDet (`Riksarkivet/rtmdet_lines`) | Object detection | [`run_htrflow.py`](run_htrflow.py) |

### Running

```bash
# YOLO only
python experiments/run_htrflow.py --model yolo

# RTMDet only
python experiments/run_htrflow.py --model rtmdet

# Both YOLO and RTMDet
python experiments/run_htrflow.py --model both

# All three models (HTRflow + Kraken)
python experiments/run_all.py

# Custom input/output
python experiments/run_all.py --folios /path/to/images --output /path/to/outputs
```

Outputs go to `outputs/htrflow_yolo/`, `outputs/htrflow_rtmdet/`, and `outputs/kraken_blla/` — visualized PNGs showing line polygon overlays on the original folios.

### htrflow YAML configs

`pipelines/yolo_pipeline.yaml` and `pipelines/rtmdet_pipeline.yaml` are declarative htrflow pipeline configs that can be run with `htrflow pipeline <yaml>`. The Python runners (`run_htrflow.py`, `run_all.py`) use the same models via the Python API instead.

### Environment setup

The full OpenMMLab stack is required for RTMDet:

```bash
conda activate line-seg-eval
pip install yapf mmengine "mmcv==2.0.1" mmdet mmocr
```

**Apple Silicon workaround:** `mmcv 2.0.1` with torch 2.x on Apple Silicon hits a missing `MPSStream::commit()` symbol. See the workaround in `run_htrflow.py` (`_prepare_rtmdet()`): a stub dylib is preloaded to satisfy the linker before the mmcv C extension loads.

---

## 2. PyLaia HTR Baselines

Zero-shot HTR baselines on 4 Gothic textualis chant manuscript folios, using pre-trained PyLaia models before any fine-tuning. See [`pylaia_baseline/README.md`](pylaia_baseline/README.md) for full details.

### Models tested

| Model | HuggingFace ID |
|---|---|
| Home Alcar | `Teklia/pylaia-home-alcar` |
| Himanis | `Teklia/pylaia-himanis` |

### Three-stage pipeline

```
01_segment.py        Kraken BLLA segmentation → segmentation JSON
02_extract_crops.py  Extract 128px-high line crops → crop PNGs
03_run_pylaia.py     PyLaia inference → transcriptions + results.csv
```

Stages 1–2 are shared across models. Stage 3 is per-model.

### Running

```bash
# Requires a separate pylaia-env conda environment (see below)
conda activate line-seg-eval
python experiments/pylaia_baseline/pylaia_home_alcar/run_experiment.py
python experiments/pylaia_baseline/pylaia_himanis/run_experiment.py
```

### PyLaia environment setup

PyLaia requires a separate conda environment due to torch version conflicts with the main `line-seg-eval` env:

```bash
conda create -n pylaia-env python=3.10 -y
conda run -n pylaia-env pip install pylaia huggingface_hub 'setuptools<72' 'torchmetrics==0.4.1'
```

The experiment scripts call `pylaia-htr-decode-ctc` from this environment as a subprocess.
