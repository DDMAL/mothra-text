# mothra-text

Experiments in automatic text/music line segmentation on medieval manuscript folios.

## Current experiment: line segmentation model comparison

Three models compared head-to-head on a set of medieval manuscript folio images:

| Model | Tool | HuggingFace ID |
|---|---|---|
| YOLOv9 lines | htrflow | `Riksarkivet/yolov9-lines-1` |
| RTMDet lines | htrflow | `Riksarkivet/rtmdet_lines` |
| BLLA baseline segmenter | Kraken | (built-in default model) |

### Repo layout

```
mothra-text/
├── data/
│   └── folios/          # original manuscript folio images
├── outputs/
│   ├── htrflow_yolo/    # annotated images — YOLO line polygons (green)
│   ├── htrflow_rtmdet/  # annotated images — RTMDet masks (blue-orange)
│   └── kraken_blla/     # annotated images — baselines (orange) + polygons (purple)
├── pipelines/
│   ├── yolo_pipeline.yaml
│   └── rtmdet_pipeline.yaml
├── run_htrflow.py       # runs YOLO and/or RTMDet via htrflow Python API
└── run_kraken.py        # runs Kraken BLLA segmenter
```

### Environment setup

```bash
conda create -n line-seg-eval python=3.10 -y
conda activate line-seg-eval

pip install htrflow kraken

# OpenMMLab stack for htrflow's RTMDet adapter
pip install yapf==0.40.1 mmengine --no-build-isolation
pip install mmcv==2.0.1 --no-build-isolation   # builds from source
pip install mmdet==3.1.0 mmocr==1.0.1
```

> **Apple Silicon note:** `mmcv 2.0.1` compiled against `torch 2.10.0` references
> `at::mps::MPSStream::commit(bool)`, a symbol removed from torch's MPS backend
> after 2.0. `run_htrflow.py` works around this at runtime by preloading a stub
> dylib (`/tmp/libmps_stub.dylib`) before importing mmcv. Build the stub once:
> ```bash
> cat > /tmp/mps_stub.cpp << 'EOF'
> namespace at { namespace mps {
> class MPSStream { public: void commit(bool); };
> void MPSStream::commit(bool) {}
> }}
> EOF
> clang++ -dynamiclib -std=c++17 -o /tmp/libmps_stub.dylib /tmp/mps_stub.cpp
> ```

### Running

```bash
# Run all three models (images from data/folios/, outputs to outputs/)
python run_all.py

# Use your own image directory (outputs still go to the repo's outputs/)
python run_all.py --folios /path/to/your/images

# Use your own image and output directories (nothing written to the repo)
python run_all.py --folios /path/to/your/images --output /path/to/your/outputs
```

Output subfolders are created automatically:
- `<output>/htrflow_yolo/`
- `<output>/htrflow_rtmdet/`
- `<output>/kraken_blla/`

Already-processed images are skipped on re-runs.

The individual scripts also accept the same flags and can be run separately:

```bash
python run_htrflow.py --model yolo
python run_htrflow.py --model rtmdet
python run_kraken.py
# all accept --folios and --output
```
