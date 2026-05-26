# mothra-text

Experiments and pipeline components for HTR and HTR-OMR alignment on medieval chant manuscripts.

## Projects

### 1. Line segmentation model comparison

Three line segmentation models compared head-to-head on a set of medieval manuscript folio images.
See the [Running](#running) section below for how to run them.

| Model | Tool | HuggingFace ID |
|---|---|---|
| YOLOv9 lines | htrflow | `Riksarkivet/yolov9-lines-1` |
| RTMDet lines | htrflow | `Riksarkivet/rtmdet_lines` |
| BLLA baseline segmenter | Kraken | (built-in default model) |

### 2. PyLaia HTR baselines

Zero-shot HTR on a 4-folio subset using two PyLaia models trained on medieval Latin manuscripts
(Teklia/pylaia-home-alcar and Teklia/pylaia-himanis). See `experiments/pylaia_baseline/`.

### 3. Pipeline Inspector GUI

A browser-based viewer for inspecting pipeline output — folio image overlaid with
Kraken line polygons and word bounding boxes, with per-layer toggles.

**Live:** https://ddmal.github.io/mothra-text/ — load any folio image + pipeline JSON,
no install required. See [`gui/README.md`](gui/README.md) for usage and how to generate
the pipeline JSON with `run_pipeline.py --export-json`.

### 4. Ground-truth-aware word segmentation

A custom HTRflow pipeline step (`steps/`) that substitutes Cantus ground-truth text for the
recognised transcription when computing word boundaries, so downstream syllable segmentation
and neume alignment use authoritative text rather than error-prone HTR output.
See [`steps/README.md`](steps/README.md).

---

## Repo layout

```
mothra-text/
├── data/folios/                    # manuscript folio images (HuggingFace)
├── experiments/
│   └── pylaia_baseline/            # zero-shot HTR baselines — see README inside
├── gui/                            # Pipeline Inspector browser app (→ ddmal.github.io/mothra-text/)
│   └── README.md
├── pipelines/                      # htrflow YAML configs for line-seg models
├── scripts/
│   └── build_gt_manifest.py        # CLI: build a Cantus gt_lookup manifest
├── steps/
│   ├── ground_truth_word_segmentation.py
│   ├── gt_manifest.py
│   └── README.md                   # word segmentation project docs
├── tests/                          # pytest suite for steps/
├── run_all.py
├── run_htrflow.py
├── run_kraken.py
└── run_pipeline.py                 # end-to-end pipeline: Kraken → PyLaia → GT word seg
```

---

## Data

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

---

## Environment setup

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

---

## Running

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
