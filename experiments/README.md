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

Outputs go to `outputs/htrflow_yolo/`, `outputs/htrflow_rtmdet/`, and `outputs/kraken_blla/`. Each folio produces two files per model:

- **`{stem}_{model}.jpg`** — visualization with line polygon overlays on the original folio
- **`{stem}_{model}.json`** — segmentation data in the following schema:

```json
{
  "folio": "stem",
  "source": "relative/path/to/image",
  "image_width": 1234,
  "image_height": 5678,
  "model_name": "yolov9_lines",
  "run_date": "2026-07-02T12:00:00+00:00",
  "lines": [
    {"id": 0, "boundary": [[x, y], ...], "baseline": null},
    ...
  ]
}
```

`baseline` is always `null` for YOLO and RTMDet (neither model predicts baselines). Kraken BLLA (`run_kraken.py`) uses the same schema and populates `baseline` when detected.

### HuggingFace output structure

Outputs are stored in `DDMAL-lab/mothra-text-outputs` under per-model folders, each split into `segmentation/` (JSON) and `visualization/` (JPG) subfolders:

```
mothra-text-outputs/
├── kraken/
│   ├── segmentation/{stem}_kraken.json
│   └── visualization/{stem}_kraken.jpg
├── htrflow-yolov9/
│   ├── segmentation/{stem}_yolo.json
│   └── visualization/{stem}_yolo.jpg
└── htrflow-rtmdet-lines/
    ├── segmentation/{stem}_rtmdet.json
    └── visualization/{stem}_rtmdet.jpg
```

### htrflow YAML configs

`pipelines/yolo_pipeline.yaml` and `pipelines/rtmdet_pipeline.yaml` are declarative htrflow pipeline configs that can be run with `htrflow pipeline <yaml>`. The Python runners (`run_htrflow.py`, `run_all.py`) use the same models via the Python API instead.

### Environment setup

The full OpenMMLab stack is required for RTMDet:

```bash
conda activate line-seg-eval
pip install yapf mmengine "mmcv==2.0.1" --no-build-isolation
pip install "mmdet==3.1.0" mmocr
```

`--no-build-isolation` is required for mmcv 2.0.1 because its build script uses `pkg_resources`, which is absent in the isolated build environment created by newer pip/setuptools.

**mmdet version pin:** mmdet ≥ 3.2.0 renamed `return_datasample` → `return_datasamples` in `DetInferencer`, breaking htrflow's RTMDet adapter. Pin to `mmdet==3.1.0`.

**Apple Silicon workaround:** `mmcv 2.0.1` with torch 2.x on Apple Silicon hits a missing `MPSStream::commit()` symbol. See the workaround in `run_htrflow.py` (`_prepare_rtmdet()`): a stub dylib is preloaded to satisfy the linker before the mmcv C extension loads.

**mmcv source-build warning:** Building mmcv from source against torch ≥ 2.6.0 headers produces a binary that references `c10::TensorImpl::decref_pyobject()`, a symbol removed from `libc10.dylib`. Use the prebuilt-equivalent `mmcv==2.0.1` (via `--no-build-isolation`) instead of building newer mmcv from source.
