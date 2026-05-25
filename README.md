# mothra-text

Experiments in automatic text/music line segmentation on medieval manuscript folios.

## Current experiment: line segmentation model comparison

Three models compared head-to-head on a set of medieval manuscript folio images:

| Model | Tool | HuggingFace ID |
|---|---|---|
| YOLOv9 lines | htrflow | `Riksarkivet/yolov9-lines-1` |
| RTMDet lines | htrflow | `Riksarkivet/rtmdet_lines` |
| BLLA baseline segmenter | Kraken | (built-in default model) |

### Data

Folio images and model outputs are stored on HuggingFace, not in this repo.
Pull them locally before running experiments:

```bash
# Pull folio images → data/folios/
ddmal-hfsync pull-groundtruth --shared --dir data

# Pull model outputs → outputs/
ddmal-hfsync pull-runs --project mothra-text --model kraken --dir outputs/kraken_blla
ddmal-hfsync pull-runs --project mothra-text --model htrflow-yolov9 --dir outputs/htrflow_yolo
ddmal-hfsync pull-runs --project mothra-text --model htrflow-rtmdet-lines --dir outputs/htrflow_rtmdet
ddmal-hfsync pull-runs --project mothra-text --model pylaia_baseline --dir outputs/pylaia_baseline
```

See [DDMAL/ddmal_hfsync](https://github.com/DDMAL/ddmal_hfsync) for setup instructions
(`~/.hfconfig` must be configured before these commands will work).

### Repo layout

```
mothra-text/
├── experiments/
│   └── pylaia_baseline/         # zero-shot HTR baselines (multiple models)
│       ├── 01_segment.py        # shared: Kraken BLLA → line coord JSON + visualisation
│       ├── 02_extract_crops.py  # shared: JSON → 128px-high grayscale line crop PNGs
│       ├── folios.txt           # shared: list of folios used in all sub-experiments
│       ├── README.md            # index of all sub-experiments
│       ├── pylaia_home_alcar/   # Teklia/pylaia-home-alcar (Latin medieval)
│       │   ├── 03_run_pylaia.py
│       │   ├── run_experiment.py
│       │   └── README.md
│       └── pylaia_himanis/      # Teklia/pylaia-himanis (French medieval)
│           ├── 03_run_pylaia.py
│           ├── run_experiment.py
│           └── README.md
├── pipelines/
│   ├── yolo_pipeline.yaml
│   └── rtmdet_pipeline.yaml
├── run_all.py           # runs all three segmentation models
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

> **Prerequisites:** `data/folios/` must be populated before running with the defaults.
> Pull it from HuggingFace first — see the [Data](#data) section above.

```bash
# Run all three models — reads from data/folios/, writes to outputs/ (both gitignored; pull from HF first)
python run_all.py

# Use a different folio directory
python run_all.py --folios /path/to/your/images

# Use a different folio directory and output location
python run_all.py --folios /path/to/your/images --output /path/to/your/outputs
```

Output subfolders are created automatically under `--output` (default: `outputs/`):
- `<output>/htrflow_yolo/`
- `<output>/htrflow_rtmdet/`
- `<output>/kraken_blla/`

Already-processed images are skipped on re-runs.

To share results with the lab, push your outputs to HuggingFace when done:

```bash
ddmal-hfsync push-run --project mothra-text --model <model> --dir outputs/<model_dir> --force
```

The individual scripts also accept the same flags and can be run separately:

```bash
python run_htrflow.py --model yolo
python run_htrflow.py --model rtmdet
python run_kraken.py
# all accept --folios and --output
```
